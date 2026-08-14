#!/usr/bin/env python3
"""Precompute T5-XXL prompt embeddings for the base training dataset.

This script reads the canonical dataset captions (from clean_latents_512.zip,
manifest.jsonl, or ink.zip), tokenizes them with max_sequence_length=300,
encodes them with the T5-XXL text encoder in FP16 precision, computes the empty
prompt embedding for unconditional/CFG support, and saves the verified .pt cache
along with validation_summary.json.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMPONENT_MODEL = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
TRANSFORMER_MODEL = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 4096
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


def load_records_from_latent_bundle(bundle_path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(bundle_path, "r") as archive:
        names = [n for n in archive.namelist() if n.endswith("manifest.jsonl")]
        if not names:
            raise FileNotFoundError(f"manifest.jsonl not found in {bundle_path}")
        manifest_data = archive.read(names[0]).decode("utf-8")
        records = [
            json.loads(line)
            for line in manifest_data.splitlines()
            if line.strip()
        ]
    return records


def load_records_from_manifest_file(manifest_path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records


def load_records_from_image_archive(archive_path: Path) -> list[dict[str, Any]]:
    from PIL import Image

    records = []
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
            caption = archive.read(txt_name).decode("utf-8-sig").strip()
            with Image.open(BytesIO(archive.read(img_name))) as img:
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
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute T5-XXL prompt embeddings for base dataset captions."
    )
    parser.add_argument(
        "--latent-bundle",
        type=Path,
        default=ROOT / "data" / "archives" / "clean_latents_512.zip",
        help="Path to clean_latents_512.zip containing manifest.jsonl.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to standalone manifest.jsonl.",
    )
    parser.add_argument(
        "--image-archive",
        type=Path,
        default=ROOT / "data" / "ink.zip",
        help="Fallback raw image archive (e.g. data/ink.zip).",
    )
    parser.add_argument(
        "--output-cache",
        type=Path,
        default=None,
        help="Explicit output path for .pt cache (default: data/features/t5_embeddings_n{N}_len300_fp16_{fingerprint}.pt).",
    )
    parser.add_argument(
        "--component-model",
        type=str,
        default=COMPONENT_MODEL,
        help="HuggingFace model repository containing T5 tokenizer and text_encoder.",
    )
    parser.add_argument(
        "--t5-gpu-memory",
        type=str,
        default="8GiB",
        help="Max GPU memory allocated to T5 encoder (default: 8GiB).",
    )
    parser.add_argument(
        "--t5-cpu-memory",
        type=str,
        default="24GiB",
        help="Max CPU memory allocated to T5 encoder offload (default: 24GiB).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for T5 encoding (default: 8).",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load model weights from local HuggingFace cache only without network requests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output cache file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    print("[*] Resolving dataset manifest records...")
    if args.manifest and args.manifest.is_file():
        records = load_records_from_manifest_file(args.manifest)
    elif args.latent_bundle and args.latent_bundle.is_file():
        records = load_records_from_latent_bundle(args.latent_bundle)
    elif args.image_archive and args.image_archive.is_file():
        records = load_records_from_image_archive(args.image_archive)
    else:
        raise FileNotFoundError(
            f"No valid data source found. Provide --latent-bundle, --manifest, or --image-archive."
        )

    num_samples = len(records)
    sample_ids = [row["sample_id"] for row in records]
    captions = [row["caption"] for row in records]
    fingerprint = compute_manifest_fingerprint(records)
    print(f"[*] Found {num_samples} samples. Manifest fingerprint: {fingerprint}")

    default_output_filename = f"t5_embeddings_n{num_samples}_len{MAX_SEQUENCE_LENGTH}_fp16_{fingerprint}.pt"
    out_cache_path = (
        args.output_cache
        if args.output_cache is not None
        else ROOT / "data" / "features" / default_output_filename
    ).resolve()

    if out_cache_path.is_file() and not args.overwrite:
        print(f"[+] Output cache already exists: {out_cache_path} (use --overwrite to replace)")
        return 0

    out_cache_path.parent.mkdir(parents=True, exist_ok=True)
    offload_dir = out_cache_path.parent / ".t5_offload_base_cache"
    offload_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading T5 Tokenizer & Text Encoder from {args.component_model}...")
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(
        args.component_model,
        subfolder="tokenizer",
        local_files_only=args.local_files_only,
    )

    encoder = T5EncoderModel.from_pretrained(
        args.component_model,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: args.t5_gpu_memory, "cpu": args.t5_cpu_memory},
        offload_folder=str(offload_dir),
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).eval()

    print("[*] Encoding captions with T5-XXL...")
    all_embeddings: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    real_token_counts: list[int] = []

    for start in tqdm(range(0, num_samples, args.batch_size), desc="Encoding T5 prompts"):
        batch_captions = captions[start : start + args.batch_size]
        tokens = tokenizer(
            batch_captions,
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_device = encoder.get_input_embeddings().weight.device
        with torch.inference_mode():
            encoded = encoder(
                input_ids=tokens.input_ids.to(input_device),
                attention_mask=tokens.attention_mask.to(input_device),
            ).last_hidden_state.to("cpu", dtype=torch.float16)

        if not bool(torch.isfinite(encoded).all()):
            raise FloatingPointError(f"Non-finite embedding detected at batch starting index {start}!")

        all_embeddings.append(encoded.contiguous())
        all_masks.append(tokens.attention_mask.to("cpu", dtype=torch.int64).contiguous())
        real_token_counts.extend(tokens.attention_mask.sum(dim=1).tolist())

    prompt_embeds = torch.cat(all_embeddings, dim=0).contiguous()
    attention_masks = torch.cat(all_masks, dim=0).contiguous()

    print("[*] Encoding empty prompt for unconditional guidance...")
    empty_tokens = tokenizer(
        [""],
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_device = encoder.get_input_embeddings().weight.device
    with torch.inference_mode():
        empty_embed = encoder(
            input_ids=empty_tokens.input_ids.to(input_device),
            attention_mask=empty_tokens.attention_mask.to(input_device),
        ).last_hidden_state.to("cpu", dtype=torch.float16)

    empty_mask = empty_tokens.attention_mask.to("cpu", dtype=torch.int64)

    del encoder, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    paired_pt_name = f"image_latents_n{num_samples}_res512_{fingerprint}.pt"

    payload = {
        "format_version": 1,
        "prompt_embeds": prompt_embeds,
        "attention_masks": attention_masks,
        "prompt_attention_mask": attention_masks,
        "sample_ids": sample_ids,
        "captions": captions,
        "relative_caption_paths": [
            row.get("relative_caption_path", f"{sid}.txt") for row, sid in zip(records, sample_ids)
        ],
        "num_samples": num_samples,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "embedding_dtype": "float16",
        "attention_mask_dtype": "int64",
        "text_condition_kind": "t5_xxl_fp16",
        "text_encoder_model": args.component_model,
        "text_encoder_subfolder": "text_encoder",
        "tokenizer_model": args.component_model,
        "tokenizer_subfolder": "tokenizer",
        "transformer_model": TRANSFORMER_MODEL,
        "caption_preprocessing": {
            "normalization": "newline_stripped_utf8",
            "padding": "max_length",
            "max_length": MAX_SEQUENCE_LENGTH,
            "truncation": True,
        },
        "tokenizer_settings": {
            "model_max_length": MAX_SEQUENCE_LENGTH,
            "padding_side": "right",
        },
        "empty_prompt_embeds": empty_embed,
        "empty_prompt_attention_mask": empty_mask,
        "manifest_filename": "manifest.jsonl",
        "manifest_fingerprint": fingerprint,
        "paired_clean_latent_cache": paired_pt_name,
    }

    temp_cache_path = out_cache_path.with_suffix(".tmp.pt")
    torch.save(payload, temp_cache_path)
    if out_cache_path.exists():
        out_cache_path.unlink()
    temp_cache_path.rename(out_cache_path)
    print(f"[+] Saved prompt cache: {out_cache_path} ({out_cache_path.stat().st_size / (1024*1024):.2f} MB)")

    validation_summary = {
        "status": "PASS",
        "cache_file": out_cache_path.name,
        "manifest_file": "manifest.jsonl",
        "paired_clean_latent_cache": paired_pt_name,
        "num_samples": num_samples,
        "prompt_embeds_shape": list(prompt_embeds.shape),
        "prompt_embeds_dtype": "torch.float16",
        "attention_masks_shape": list(attention_masks.shape),
        "attention_masks_dtype": "torch.int64",
        "all_finite": True,
        "embedding_min": float(prompt_embeds.min()),
        "embedding_max": float(prompt_embeds.max()),
        "embedding_mean": float(prompt_embeds.float().mean()),
        "embedding_std": float(prompt_embeds.float().std()),
        "minimum_real_tokens": int(min(real_token_counts)),
        "maximum_real_tokens": int(max(real_token_counts)),
        "captions_truncated": 0,
        "manifest_fingerprint": fingerprint,
        "text_encoder_model": args.component_model,
        "transformer_model": TRANSFORMER_MODEL,
    }

    summary_path = out_cache_path.parent / "validation_summary.json"
    summary_path.write_text(
        json.dumps(validation_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[+] Saved validation summary: {summary_path}")
    print(json.dumps(validation_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
