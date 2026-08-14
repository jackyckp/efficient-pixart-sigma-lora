#!/usr/bin/env python3
"""Precompute SDXL VAE clean latents and build a latent bundle archive.

This script preprocesses raw images (aspect-ratio-preserving 512x512 center crop,
normalized to [-1, 1]), encodes them into clean FP16 latents using the SDXL VAE
bundled with PixArt-Sigma, creates a deterministic manifest, and optionally
packages the artifacts into a verified ZIP archive (e.g., clean_latents_512.zip).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image, ImageOps
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMPONENT_MODEL = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
TRANSFORMER_MODEL = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
DEFAULT_RESOLUTION = 512
DEFAULT_SCALING_FACTOR = 0.13025
EXPECTED_DATASET_SIZE = 260
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def natural_key(path_str: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path_str.lower().replace("\\", "/"))
    return tuple(int(p) if p.isdigit() else p for p in parts)


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def compute_manifest_fingerprint(manifest: Sequence[dict[str, Any]]) -> str:
    payload = [
        (
            row["sample_id"],
            normalize_newlines(row["caption"]),
            row["original_width"],
            row["original_height"],
        )
        for row in manifest
    ]
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:12]


def preprocess_pil(image: Image.Image, resolution: int = 512) -> Image.Image:
    return ImageOps.fit(
        image.convert("RGB"),
        (resolution, resolution),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_dataset_from_archive(
    archive_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    records: list[dict[str, Any]] = []
    image_bytes_map: dict[str, bytes] = {}

    with zipfile.ZipFile(archive_path, "r") as archive:
        names = sorted(
            [
                n
                for n in archive.namelist()
                if not n.endswith("/")
                and PurePosixPath(n).suffix.lower() in SUPPORTED_EXTS
            ],
            key=natural_key,
        )

        for img_name in names:
            txt_name = PurePosixPath(img_name).with_suffix(".txt").as_posix()
            try:
                caption = archive.read(txt_name).decode("utf-8-sig").strip()
            except KeyError:
                raise FileNotFoundError(
                    f"Missing caption {txt_name} for image {img_name} in {archive_path}"
                )

            img_bytes = archive.read(img_name)
            with Image.open(BytesIO(img_bytes)) as img:
                width, height = img.size

            p = PurePosixPath(img_name)
            parts = p.parts
            if parts and parts[0] == "ink":
                sample_id = "/".join(parts[1:]).rsplit(".", 1)[0]
                rel_img = "/".join(parts[1:])
                rel_txt = rel_img.rsplit(".", 1)[0] + ".txt"
            else:
                sample_id = p.as_posix().rsplit(".", 1)[0]
                rel_img = p.as_posix()
                rel_txt = rel_img.rsplit(".", 1)[0] + ".txt"

            image_bytes_map[sample_id] = img_bytes
            records.append(
                {
                    "sample_id": sample_id,
                    "caption": normalize_newlines(caption),
                    "original_width": width,
                    "original_height": height,
                    "relative_image_path": rel_img,
                    "relative_caption_path": rel_txt,
                }
            )

    return records, image_bytes_map


def load_dataset_from_dir(
    dir_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    records: list[dict[str, Any]] = []
    image_paths_map: dict[str, Path] = {}

    all_images = sorted(
        [
            p
            for p in dir_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ],
        key=lambda p: natural_key(p.as_posix()),
    )

    paired_images = [p for p in all_images if p.with_suffix(".txt").is_file()]
    if not paired_images:
        raise RuntimeError(f"No paired images and captions found under {dir_path}")

    dataset_root = Path(os.path.commonpath([str(p.parent) for p in paired_images]))
    for img_path in paired_images:
        txt_path = img_path.with_suffix(".txt")
        caption = txt_path.read_text(encoding="utf-8-sig").strip()
        with Image.open(img_path) as img:
            width, height = img.size

        rel_img = img_path.relative_to(dataset_root).as_posix()
        rel_txt = txt_path.relative_to(dataset_root).as_posix()
        sample_id = rel_img.rsplit(".", 1)[0]

        image_paths_map[sample_id] = img_path
        records.append(
            {
                "sample_id": sample_id,
                "caption": normalize_newlines(caption),
                "original_width": width,
                "original_height": height,
                "relative_image_path": rel_img,
                "relative_caption_path": rel_txt,
            }
        )

    return records, image_paths_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute SDXL VAE clean latents and package clean_latents_512.zip."
    )
    parser.add_argument(
        "--image-archive",
        type=Path,
        default=ROOT / "data" / "ink.zip",
        help="Path to raw image zip archive (e.g. data/ink.zip).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional directory containing raw images and .txt captions.",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=ROOT / "data" / "archives" / "clean_latents_512.zip",
        help="Path to output zip bundle (e.g. data/archives/clean_latents_512.zip).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "archives" / "clean_latents_512",
        help="Temporary or direct output directory for manifest and tensors.",
    )
    parser.add_argument(
        "--component-model",
        type=str,
        default=COMPONENT_MODEL,
        help="HuggingFace model repository containing the SDXL VAE.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Image resolution for center-cropping (default: 512).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for VAE encoding (default: 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for latent sampling (default: 42).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output zip or files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.output_zip and args.output_zip.exists() and not args.overwrite:
        print(f"Output archive already exists: {args.output_zip} (use --overwrite to replace)")
        return 0

    print("[*] Loading image and caption records...")
    if args.image_dir and args.image_dir.is_dir():
        records, image_sources = load_dataset_from_dir(args.image_dir)
        is_archive = False
    elif args.image_archive and args.image_archive.is_file():
        records, image_sources = load_dataset_from_archive(args.image_archive)
        is_archive = True
    else:
        raise FileNotFoundError(
            f"Neither image directory ({args.image_dir}) nor image archive ({args.image_archive}) was found."
        )

    num_samples = len(records)
    sample_ids = [row["sample_id"] for row in records]
    fingerprint = compute_manifest_fingerprint(records)
    print(f"[*] Found {num_samples} samples. Manifest fingerprint: {fingerprint}")

    print(f"[*] Loading SDXL VAE from {args.component_model}/vae on {args.device}...")
    vae = AutoencoderKL.from_pretrained(
        args.component_model,
        subfolder="vae",
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    ).eval().to(args.device)
    vae.requires_grad_(False)

    scaling_factor = float(vae.config.scaling_factor)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_size = args.resolution // vae_scale_factor

    print(f"[*] Scaling factor: {scaling_factor}, Latent size: {latent_size}x{latent_size}")
    print("[*] Encoding clean latents...")

    all_latents: list[torch.Tensor] = []
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    for start in tqdm(range(0, num_samples, args.batch_size), desc="Encoding VAE latents"):
        batch_records = records[start : start + args.batch_size]
        batch_pixels: list[torch.Tensor] = []

        for row in batch_records:
            sid = row["sample_id"]
            if is_archive:
                with Image.open(BytesIO(image_sources[sid])) as pil_img:
                    cropped = preprocess_pil(pil_img, resolution=args.resolution)
            else:
                with Image.open(image_sources[sid]) as pil_img:
                    cropped = preprocess_pil(pil_img, resolution=args.resolution)
            batch_pixels.append(pil_to_tensor(cropped))

        pixels = torch.stack(batch_pixels).to(
            args.device,
            dtype=torch.float16 if args.device == "cuda" else torch.float32,
        )

        with torch.inference_mode():
            posterior = vae.encode(pixels).latent_dist
            batch_latents = posterior.sample(generator=generator) * scaling_factor

        all_latents.append(batch_latents.cpu().to(torch.float16).contiguous())
        del pixels, posterior, batch_latents

    clean_latents = torch.cat(all_latents, dim=0).contiguous()
    del vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    assert clean_latents.shape == (num_samples, 4, latent_size, latent_size)
    assert bool(torch.isfinite(clean_latents).all()), "Non-finite latents detected!"

    manifest_lines = [
        json.dumps(
            {
                "sample_id": row["sample_id"],
                "relative_image_path": row["relative_image_path"],
                "relative_caption_path": row["relative_caption_path"],
                "caption": row["caption"],
                "original_width": row["original_width"],
                "original_height": row["original_height"],
            },
            ensure_ascii=False,
        )
        + "\n"
        for row in records
    ]

    pt_filename = f"image_latents_n{num_samples}_res{args.resolution}_{fingerprint}.pt"
    latent_cache = {
        "format_version": 1,
        "latents": clean_latents,
        "sample_ids": sample_ids,
        "relative_image_paths": [row["relative_image_path"] for row in records],
        "image_paths": [row["relative_image_path"] for row in records],
        "num_images": num_samples,
        "resolution": args.resolution,
        "latent_shape_per_image": [4, latent_size, latent_size],
        "latent_dtype": "float16",
        "latent_kind": "clean_x0_scaled",
        "vae_model": args.component_model,
        "vae_subfolder": "vae",
        "transformer_model": TRANSFORMER_MODEL,
        "scaling_factor": scaling_factor,
        "vae_scaling_factor": scaling_factor,
        "vae_scale_factor": vae_scale_factor,
        "latent_sampling": "posterior.sample",
        "seed": args.seed,
        "preprocessing": {
            "convert": "RGB",
            "resize_crop": "PIL.ImageOps.fit",
            "size": [args.resolution, args.resolution],
            "resampling": "LANCZOS",
            "centering": [0.5, 0.5],
            "normalization": "pixel / 127.5 - 1.0",
        },
        "manifest_filename": "manifest.jsonl",
        "manifest_fingerprint": fingerprint,
    }

    validation_summary = {
        "status": "PASS",
        "cache_file": pt_filename,
        "manifest_file": "manifest.jsonl",
        "num_images": num_samples,
        "latents_shape": list(clean_latents.shape),
        "latents_dtype": "torch.float16",
        "all_finite": True,
        "latent_min": float(clean_latents.min()),
        "latent_max": float(clean_latents.max()),
        "latent_mean": float(clean_latents.float().mean()),
        "latent_std": float(clean_latents.float().std()),
        "scaling_factor": scaling_factor,
        "manifest_fingerprint": fingerprint,
    }

    # Write staging files
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.write_text("".join(manifest_lines), encoding="utf-8")

    pt_path = out_dir / pt_filename
    torch.save(latent_cache, pt_path)

    summary_path = out_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(validation_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[+] Saved directory assets to: {out_dir}")

    # Package into output zip if requested
    if args.output_zip:
        zip_path = args.output_zip.resolve()
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        temp_zip = zip_path.with_suffix(".tmp.zip")

        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_path, arcname="manifest.jsonl")
            zf.write(summary_path, arcname="validation_summary.json")
            zf.write(pt_path, arcname=pt_filename)

        if zip_path.exists():
            zip_path.unlink()
        temp_zip.rename(zip_path)
        print(f"[+] Successfully packaged latent bundle: {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")

    print(json.dumps(validation_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
