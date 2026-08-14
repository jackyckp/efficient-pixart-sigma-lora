#!/usr/bin/env python3
"""Evaluate saved teacher/student image sets with CLIP, CMMD and latency."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import zipfile
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont

from scripts.distillation.common import repository_root, write_json


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description="Compare matched 20-step teacher and distilled images."
    )
    parser.add_argument("--teacher-images", type=Path, required=True)
    parser.add_argument("--student-images", type=Path, required=True)
    parser.add_argument(
        "--evaluation-prompts",
        type=Path,
        default=root / "evaluation" / "distillation_prompts_v1.json",
    )
    parser.add_argument(
        "--image-archive",
        type=Path,
        default=(
            root / "data" / "ink.zip"
            if (root / "data" / "ink.zip").is_file()
            else root / "data" / "archives" / "ink.zip"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--cmmd-bandwidth", type=float, default=10.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def gaussian_rbf_kernel(
    left: torch.Tensor,
    right: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    if bandwidth <= 0 or not math.isfinite(bandwidth):
        raise ValueError("CMMD bandwidth must be finite and positive.")
    distances = torch.cdist(left.float(), right.float()).square()
    return torch.exp(-distances / (2.0 * bandwidth * bandwidth))


def unbiased_cmmd(
    left: torch.Tensor,
    right: torch.Tensor,
    bandwidth: float = 10.0,
) -> float:
    """Squared unbiased MMD on normalized CLIP image embeddings."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("CMMD inputs must be compatible rank-2 embeddings.")
    if len(left) < 2 or len(right) < 2:
        raise ValueError("CMMD requires at least two samples per set.")
    xx = gaussian_rbf_kernel(left, left, bandwidth)
    yy = gaussian_rbf_kernel(right, right, bandwidth)
    xy = gaussian_rbf_kernel(left, right, bandwidth)
    term_x = (xx.sum() - xx.diagonal().sum()) / (len(left) * (len(left) - 1))
    term_y = (yy.sum() - yy.diagonal().sum()) / (
        len(right) * (len(right) - 1)
    )
    value = term_x + term_y - 2.0 * xy.mean()
    return float(value.item())


def _load_prompt_manifest(path: Path) -> tuple[dict[str, Any], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Evaluation prompt manifest has no prompts.")
    return tuple(prompts)


def expected_image_records(
    prompt_records: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for prompt in prompt_records:
        for seed in prompt["seeds"]:
            records.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "prompt": prompt["prompt"],
                    "seed": int(seed),
                    "filename": f"{prompt['prompt_id']}_seed{int(seed)}.png",
                }
            )
    return tuple(records)


def _load_images(
    directory: Path,
    records: Sequence[dict[str, Any]],
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    images: list[Image.Image] = []
    metadata: list[dict[str, Any]] = []
    for record in records:
        path = directory / record["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation image: {path}")
        with Image.open(path) as image:
            converted = image.convert("RGB")
            if converted.size != (512, 512):
                raise ValueError(f"Evaluation image is not 512x512: {path}")
            images.append(converted.copy())
        json_path = path.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing image metadata: {json_path}")
        item = json.loads(json_path.read_text(encoding="utf-8"))
        if item.get("status") != "PASS":
            raise ValueError(f"Image metadata is not PASS: {json_path}")
        metadata.append(item)
    return images, metadata


def _load_real_plant_images(path: Path) -> list[Image.Image]:
    images: list[Image.Image] = []
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if "/plant/" in f"/{name.lower()}" and name.lower().endswith(".jpg")
        )
        for name in names:
            with Image.open(io.BytesIO(archive.read(name))) as image:
                images.append(image.convert("RGB").resize((512, 512)).copy())
    if len(images) != 209:
        raise ValueError(f"Expected 209 real plant images, got {len(images)}.")
    return images


def _clip_features_and_scores(
    *,
    teacher_images: Sequence[Image.Image],
    student_images: Sequence[Image.Image],
    real_images: Sequence[Image.Image],
    prompts: Sequence[str],
    model_name: str,
    batch_size: int,
    device: str,
    local_files_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float], list[float]]:
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    model = CLIPModel.from_pretrained(
        model_name, local_files_only=local_files_only
    ).to(device).eval()

    def image_features(images: Sequence[Image.Image]) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start in range(0, len(images), batch_size):
            batch = processor(
                images=list(images[start : start + batch_size]),
                return_tensors="pt",
            )
            with torch.inference_mode():
                features = model.get_image_features(
                    pixel_values=batch.pixel_values.to(device)
                )
            outputs.append(F_normalize(features).cpu())
        return torch.cat(outputs)

    teacher_features = image_features(teacher_images)
    student_features = image_features(student_images)
    real_features = image_features(real_images)
    text_outputs: list[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        batch = processor(
            text=list(prompts[start : start + batch_size]),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            features = model.get_text_features(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
            )
        text_outputs.append(F_normalize(features).cpu())
    text_features = torch.cat(text_outputs)
    teacher_scores = (teacher_features * text_features).sum(dim=1).tolist()
    student_scores = (student_features * text_features).sum(dim=1).tolist()
    return (
        teacher_features,
        student_features,
        real_features,
        teacher_scores,
        student_scores,
    )


def F_normalize(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(tensor.float(), dim=-1)


def _make_grids(
    output_dir: Path,
    records: Sequence[dict[str, Any]],
    teacher_images: Sequence[Image.Image],
    student_images: Sequence[Image.Image],
    limit: int = 10,
) -> list[str]:
    font = ImageFont.load_default(size=18)
    paths: list[str] = []
    seen: set[str] = set()
    for record, teacher, student in zip(records, teacher_images, student_images):
        if record["prompt_id"] in seen or len(seen) >= limit:
            continue
        seen.add(record["prompt_id"])
        canvas = Image.new("RGB", (1024, 580), "white")
        canvas.paste(teacher, (0, 68))
        canvas.paste(student, (512, 68))
        draw = ImageDraw.Draw(canvas)
        draw.text((190, 18), "20-step teacher", fill="black", font=font)
        draw.text((700, 18), "distilled student", fill="black", font=font)
        caption = f"{record['prompt_id']} | seed {record['seed']}"
        draw.text((20, 548), caption, fill="black", font=font)
        path = output_dir / f"comparison_{record['prompt_id']}.png"
        canvas.save(path)
        paths.append(str(path))
    return paths


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.clip_batch_size <= 0:
        raise ValueError("--clip-batch-size must be positive.")
    prompt_records = _load_prompt_manifest(args.evaluation_prompts.resolve())
    records = expected_image_records(prompt_records)
    teacher_images, teacher_metadata = _load_images(
        args.teacher_images.resolve(), records
    )
    student_images, student_metadata = _load_images(
        args.student_images.resolve(), records
    )
    if any(item.get("transformer_forward_calls") != 20 for item in teacher_metadata):
        raise ValueError("Teacher metadata must record exactly 20 forward calls.")
    student_step_counts = {item.get("num_inference_steps") for item in student_metadata}
    if len(student_step_counts) != 1:
        raise ValueError("Student image set mixes inference step counts.")
    student_steps = int(next(iter(student_step_counts)))
    if any(
        item.get("transformer_forward_calls") != student_steps
        or item.get("guidance_scale") != 1.0
        or item.get("classifier_free_guidance_branch") is not False
        for item in student_metadata
    ):
        raise ValueError("Student metadata violates exact-call CFG=1 contract.")
    real_images = _load_real_plant_images(args.image_archive.resolve())
    prompts = [record["prompt"] for record in records]
    (
        teacher_features,
        student_features,
        real_features,
        teacher_scores,
        student_scores,
    ) = _clip_features_and_scores(
        teacher_images=teacher_images,
        student_images=student_images,
        real_images=real_images,
        prompts=prompts,
        model_name=args.clip_model,
        batch_size=args.clip_batch_size,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    teacher_clip = statistics.fmean(teacher_scores)
    student_clip = statistics.fmean(student_scores)
    teacher_cmmd = unbiased_cmmd(
        teacher_features, real_features, args.cmmd_bandwidth
    )
    student_cmmd = unbiased_cmmd(
        student_features, real_features, args.cmmd_bandwidth
    )
    teacher_latency = statistics.median(
        float(item["denoise_seconds"]) for item in teacher_metadata
    )
    student_latency = statistics.median(
        float(item["denoise_seconds"]) for item in student_metadata
    )
    clip_ratio = student_clip / teacher_clip if teacher_clip > 0 else math.nan
    cmmd_limit = teacher_cmmd * 1.5
    speedup = teacher_latency / student_latency
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_image_clip_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("prompt_id", "seed", "teacher_clip", "student_clip"),
        )
        writer.writeheader()
        for record, teacher_score, student_score in zip(
            records, teacher_scores, student_scores
        ):
            writer.writerow(
                {
                    "prompt_id": record["prompt_id"],
                    "seed": record["seed"],
                    "teacher_clip": teacher_score,
                    "student_clip": student_score,
                }
            )
    grids = _make_grids(
        output_dir, records, teacher_images, student_images
    )
    gates = {
        "finite_metrics": all(
            math.isfinite(value)
            for value in (
                teacher_clip,
                student_clip,
                teacher_cmmd,
                student_cmmd,
                teacher_latency,
                student_latency,
            )
        ),
        "clip_at_least_90_percent": clip_ratio >= 0.9,
        "cmmd_at_most_1_5x_teacher": student_cmmd <= cmmd_limit,
        "median_latency_speedup_at_least_5x": speedup >= 5.0,
        "exact_student_forward_calls": True,
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "student_steps": student_steps,
        "image_count_per_model": len(records),
        "clip_model": args.clip_model,
        "cmmd_definition": "unbiased squared MMD, normalized CLIP image embeddings, Gaussian RBF",
        "cmmd_bandwidth": args.cmmd_bandwidth,
        "teacher_mean_clip": teacher_clip,
        "student_mean_clip": student_clip,
        "student_teacher_clip_ratio": clip_ratio,
        "teacher_to_real_cmmd": teacher_cmmd,
        "student_to_real_cmmd": student_cmmd,
        "cmmd_acceptance_limit": cmmd_limit,
        "teacher_median_denoise_seconds": teacher_latency,
        "student_median_denoise_seconds": student_latency,
        "latency_speedup": speedup,
        "gates": gates,
        "comparison_grids": grids,
    }
    write_json(output_dir / "evaluation_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
