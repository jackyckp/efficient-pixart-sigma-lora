#!/usr/bin/env python3
"""Build Base/Teacher/4-step/2-step presentation comparison candidates."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.distillation.common import (  # noqa: E402
    COMPONENT_MODEL,
    TRANSFORMER_MODEL,
    write_json,
)
from scripts.distillation.evaluation_prompt_cache import (  # noqa: E402
    DEFAULT_CACHE,
    load_evaluation_prompt_cache,
)
from scripts.distillation.generate_evaluation_set_impl import (  # noqa: E402
    _generate_teacher_final,
)


DEFAULT_FILENAMES = (
    "eval-11_seed10111.png",
    "eval-11_seed10114.png",
    "eval-12_seed10122.png",
    "eval-13_seed10132.png",
    "eval-15_seed10151.png",
    "eval-16_seed10161.png",
    "eval-16_seed10163.png",
    "eval-28_seed10281.png",
    "eval-03_seed10033.png",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate official PixArt references and four-way grids."
    )
    parser.add_argument("--filenames", nargs="+", default=list(DEFAULT_FILENAMES))
    parser.add_argument(
        "--teacher-images",
        type=Path,
        default=ROOT
        / "outputs"
        / "distillation"
        / "teammate_plant209_step4000"
        / "evaluation_4step"
        / "images"
        / "teacher",
    )
    parser.add_argument(
        "--four-step-images",
        type=Path,
        default=ROOT
        / "outputs"
        / "distillation_experiments"
        / "teacher_b_extend6k_then2step"
        / "evaluation_4step"
        / "images"
        / "student",
    )
    parser.add_argument(
        "--two-step-images",
        type=Path,
        default=ROOT
        / "outputs"
        / "distillation_experiments"
        / "teacher_b_extend6k_then2step"
        / "evaluation_2step"
        / "images"
        / "student",
    )
    parser.add_argument("--evaluation-prompt-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--empty-prompt-cache",
        type=Path,
        default=ROOT
        / "data"
        / "features"
        / "t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "outputs"
        / "evaluation"
        / "teacher_b_extend6k_fourway_candidates",
    )
    parser.add_argument("--transformer-model", default=TRANSFORMER_MODEL)
    parser.add_argument("--component-model", default=COMPONENT_MODEL)
    parser.add_argument("--official-guidance-scale", type=float, default=1.5)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _read_metadata(image_path: Path) -> dict[str, Any]:
    metadata_path = image_path.with_suffix(".json")
    if not image_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing image or metadata: {image_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "PASS":
        raise ValueError(f"Image metadata is not PASS: {metadata_path}")
    return metadata


def _load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if len(args.filenames) != len(set(args.filenames)):
        raise ValueError("Candidate filenames must be unique.")
    records: list[dict[str, Any]] = []
    contracts = (
        (args.teacher_images.resolve(), 20, 20),
        (args.four_step_images.resolve(), 4, 4),
        (args.two_step_images.resolve(), 2, 2),
    )
    for filename in args.filenames:
        if Path(filename).name != filename or not filename.lower().endswith(".png"):
            raise ValueError(f"Unsafe or non-PNG filename: {filename}")
        metadata_by_role: list[dict[str, Any]] = []
        for directory, steps, calls in contracts:
            metadata = _read_metadata(directory / filename)
            if (
                metadata.get("num_inference_steps") != steps
                or metadata.get("transformer_forward_calls") != calls
            ):
                raise ValueError(f"Inference contract mismatch: {directory / filename}")
            metadata_by_role.append(metadata)
        identity = (
            metadata_by_role[0]["prompt_id"],
            metadata_by_role[0]["prompt"],
            int(metadata_by_role[0]["seed"]),
        )
        for metadata in metadata_by_role[1:]:
            if (
                metadata["prompt_id"],
                metadata["prompt"],
                int(metadata["seed"]),
            ) != identity:
                raise ValueError(f"Prompt/seed mismatch across models: {filename}")
        records.append(
            {
                "filename": filename,
                "prompt_id": identity[0],
                "prompt": identity[1],
                "seed": identity[2],
            }
        )
    return records


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidate = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    if candidate.is_file():
        return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _make_grid(
    *,
    record: dict[str, Any],
    sources: Sequence[tuple[str, Path]],
    output_path: Path,
) -> None:
    cell = 512
    gap = 18
    margin = 24
    label_h = 62
    prompt_h = 92
    width = margin * 2 + cell * 4 + gap * 3
    height = margin + label_h + cell + prompt_h + margin
    canvas = Image.new("RGB", (width, height), "#F8F5ED")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(25, bold=True)
    prompt_font = _font(21)
    border = "#17365D"
    for index, (label, image_path) in enumerate(sources):
        x = margin + index * (cell + gap)
        bbox = draw.textbbox((0, 0), label, font=title_font)
        draw.text(
            (x + (cell - (bbox[2] - bbox[0])) / 2, margin + 12),
            label,
            fill="#092958",
            font=title_font,
        )
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, margin + label_h))
        draw.rectangle(
            (x, margin + label_h, x + cell - 1, margin + label_h + cell - 1),
            outline=border,
            width=3,
        )
    prompt_text = f"{record['prompt_id']} · seed {record['seed']} — {record['prompt']}"
    lines = textwrap.wrap(prompt_text, width=150)
    y = margin + label_h + cell + 16
    for line in lines[:2]:
        bbox = draw.textbbox((0, 0), line, font=prompt_font)
        draw.text(
            ((width - (bbox[2] - bbox[0])) / 2, y),
            line,
            fill="#222222",
            font=prompt_font,
        )
        y += 29
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for official-base generation.")
    if args.official_guidance_scale <= 1.0:
        raise ValueError("Official comparison must use CFG guidance greater than 1.0.")
    records = _load_records(args)
    features = load_evaluation_prompt_cache(
        args.evaluation_prompt_cache,
        records,
        component_model=args.component_model,
    )
    empty_cache = torch.load(
        args.empty_prompt_cache.resolve(), map_location="cpu", weights_only=True
    )
    empty_embed = empty_cache["empty_prompt_embeds"]
    empty_mask = empty_cache["empty_prompt_attention_mask"]
    if empty_embed.shape != (1, 300, 4096) or empty_mask.shape != (1, 300):
        raise ValueError("Canonical empty-prompt cache shape mismatch.")

    output_dir = args.output_dir.resolve()
    official_dir = output_dir / "official_base"
    grid_dir = output_dir / "grids"
    official_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)

    missing_records = [
        record
        for record in records
        if args.overwrite
        or not (official_dir / record["filename"]).is_file()
        or not (official_dir / record["filename"]).with_suffix(".json").is_file()
    ]
    if missing_records:
        from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, PixArtTransformer2DModel
        from diffusers.image_processor import VaeImageProcessor

        device = torch.device("cuda")
        transformer = PixArtTransformer2DModel.from_pretrained(
            args.transformer_model,
            subfolder="transformer",
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=args.local_files_only,
        ).to(device).eval()
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            args.component_model,
            subfolder="scheduler",
            local_files_only=args.local_files_only,
        )
        results: list[tuple[dict[str, Any], torch.Tensor, float]] = []
        for record in missing_records:
            embed, mask = features[record["prompt_id"]]
            torch.cuda.synchronize()
            started = time.perf_counter()
            latent = _generate_teacher_final(
                transformer=transformer,
                scheduler=scheduler,
                prompt_embed=embed.to(device),
                prompt_mask=mask.to(device),
                empty_embed=empty_embed.to(device),
                empty_mask=empty_mask.to(device),
                seed=record["seed"],
                guidance_scale=args.official_guidance_scale,
                device=device,
            )
            torch.cuda.synchronize()
            results.append((record, latent, time.perf_counter() - started))
        del transformer
        gc.collect()
        torch.cuda.empty_cache()

        vae = AutoencoderKL.from_pretrained(
            args.component_model,
            subfolder="vae",
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=args.local_files_only,
        ).to(device).eval()
        processor = VaeImageProcessor(vae_scale_factor=8)
        for record, latent, seconds in results:
            with torch.inference_mode():
                decoded = vae.decode(
                    latent.to(device) / vae.config.scaling_factor,
                    return_dict=False,
                )[0]
            image = processor.postprocess(decoded, output_type="pil")[0]
            image_path = official_dir / record["filename"]
            image.save(image_path)
            write_json(
                image_path.with_suffix(".json"),
                {
                    "status": "PASS",
                    **record,
                    "model_role": "official_pixart_base",
                    "transformer_model": args.transformer_model,
                    "num_inference_steps": 20,
                    "guidance_scale": args.official_guidance_scale,
                    "classifier_free_guidance_branch": True,
                    "transformer_forward_calls": 20,
                    "denoise_seconds": seconds,
                },
            )
        del vae
        gc.collect()
        torch.cuda.empty_cache()

    candidates: list[dict[str, Any]] = []
    for record in records:
        stem = Path(record["filename"]).stem
        grid_path = grid_dir / f"{stem}_fourway.png"
        sources = (
            ("Official Base · 20-step", official_dir / record["filename"]),
            ("Style Teacher B · 20-step", args.teacher_images.resolve() / record["filename"]),
            ("Primary Distilled · 4-step", args.four_step_images.resolve() / record["filename"]),
            ("Primary Distilled · 2-step", args.two_step_images.resolve() / record["filename"]),
        )
        _make_grid(record=record, sources=sources, output_path=grid_path)
        candidates.append(
            {
                **record,
                "grid": str(grid_path),
                "official_base": str(sources[0][1]),
                "teacher_20step": str(sources[1][1]),
                "student_4step": str(sources[2][1]),
                "student_2step": str(sources[3][1]),
            }
        )
    result = {
        "format_version": 1,
        "status": "PASS",
        "comparison_order": [
            "official_base_20step",
            "style_teacher_b_20step",
            "primary_distilled_4step",
            "primary_distilled_2step",
        ],
        "official_guidance_scale": args.official_guidance_scale,
        "student_guidance_scale": 1.0,
        "evaluation_prompt_cache": str(args.evaluation_prompt_cache.resolve()),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    write_json(output_dir / "candidate_manifest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    generate(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
