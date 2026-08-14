#!/usr/bin/env python3
"""Summarize saved 30-prompt x 4-seed CLIP scores without model inference."""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.distillation.evaluate_distilled_impl import F_normalize, unbiased_cmmd


DEFAULT_ROOT = (
    ROOT / "outputs" / "distillation_experiments" / "teacher_b_extend6k_then2step"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--four-step-csv",
        type=Path,
        default=DEFAULT_ROOT / "evaluation_4step" / "metrics" / "per_image_clip_scores.csv",
    )
    parser.add_argument(
        "--two-step-csv",
        type=Path,
        default=DEFAULT_ROOT / "evaluation_2step" / "metrics" / "per_image_clip_scores.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "evaluation" / "seed_variance_primary_models",
    )
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
        default=DEFAULT_ROOT / "evaluation_4step" / "images" / "student",
    )
    parser.add_argument(
        "--two-step-images",
        type=Path,
        default=DEFAULT_ROOT / "evaluation_2step" / "images" / "student",
    )
    parser.add_argument(
        "--image-archive",
        type=Path,
        default=ROOT / "data" / "archives" / "ink.zip",
    )
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--cmmd-bandwidth", type=float, default=10.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def _load(path: Path) -> list[dict[str, Any]]:
    with path.resolve().open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 120:
        raise ValueError(f"Expected 120 saved scores in {path}, got {len(rows)}")
    return [
        {
            "prompt_id": row["prompt_id"],
            "seed": int(row["seed"]),
            "teacher_clip": float(row["teacher_clip"]),
            "student_clip": float(row["student_clip"]),
        }
        for row in rows
    ]


def _replicate_statistics(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_id"]].append(row)
    if len(by_prompt) != 30 or any(len(values) != 4 for values in by_prompt.values()):
        raise ValueError("Expected exactly 30 prompts with four seeds per prompt.")
    replicate_means: list[float] = []
    within_prompt_variances: list[float] = []
    for prompt_rows in by_prompt.values():
        ordered = sorted(prompt_rows, key=lambda item: item["seed"])
        values = [item[field] for item in ordered]
        within_prompt_variances.append(statistics.variance(values))
    for replicate in range(4):
        replicate_means.append(
            statistics.fmean(
                sorted(values, key=lambda item: item["seed"])[replicate][field]
                for values in by_prompt.values()
            )
        )
    return {
        "mean_clip_120_images": statistics.fmean(row[field] for row in rows),
        "seed_replicate_mean_clip": statistics.fmean(replicate_means),
        "seed_replicate_sample_sd": statistics.stdev(replicate_means),
        "seed_replicate_sample_variance": statistics.variance(replicate_means),
        "mean_within_prompt_seed_variance": statistics.fmean(within_prompt_variances),
        "rms_within_prompt_seed_sd": statistics.fmean(within_prompt_variances) ** 0.5,
        "replicate_1_mean": replicate_means[0],
        "replicate_2_mean": replicate_means[1],
        "replicate_3_mean": replicate_means[2],
        "replicate_4_mean": replicate_means[3],
    }


def _image_records(rows: list[dict[str, Any]]) -> list[tuple[str, int, str]]:
    return [
        (row["prompt_id"], row["seed"], f"{row['prompt_id']}_seed{row['seed']}.png")
        for row in rows
    ]


def _load_generated_images(
    directory: Path, records: list[tuple[str, int, str]]
) -> tuple[list[Image.Image], list[float]]:
    images: list[Image.Image] = []
    latencies: list[float] = []
    for _, _, filename in records:
        path = directory.resolve() / filename
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        latencies.append(float(metadata["denoise_seconds"]))
    return images, latencies


def _load_real_images(path: Path) -> list[Image.Image]:
    images: list[Image.Image] = []
    with zipfile.ZipFile(path.resolve()) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if "/plant/" in f"/{name.lower()}" and name.lower().endswith(".jpg")
        )
        for name in names:
            with Image.open(io.BytesIO(archive.read(name))) as image:
                images.append(image.convert("RGB").resize((512, 512)).copy())
    if len(images) != 209:
        raise ValueError(f"Expected 209 real plant images, got {len(images)}")
    return images


def _feature_cache(
    args: argparse.Namespace,
    records: list[tuple[str, int, str]],
) -> tuple[dict[str, torch.Tensor], dict[str, list[float]]]:
    output_dir = args.output_dir.resolve()
    cache_path = output_dir / "clip_image_feature_cache.pt"
    roles = {
        "20-step Teacher B": args.teacher_images,
        "Primary 4-step": args.four_step_images,
        "Primary 2-step": args.two_step_images,
    }
    expected = {
        "format_version": 1,
        "clip_model": args.clip_model,
        "records": records,
        "roles": {name: str(path.resolve()) for name, path in roles.items()},
        "image_archive": str(args.image_archive.resolve()),
    }
    if cache_path.is_file():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if all(cache.get(key) == value for key, value in expected.items()):
            return cache["features"], cache["latencies"]

    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(
        args.clip_model, local_files_only=args.local_files_only
    )
    model = CLIPModel.from_pretrained(
        args.clip_model, local_files_only=args.local_files_only
    ).to(args.device).eval()

    def encode(images: list[Image.Image]) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start in range(0, len(images), args.clip_batch_size):
            batch = processor(
                images=images[start : start + args.clip_batch_size],
                return_tensors="pt",
            )
            with torch.inference_mode():
                value = model.get_image_features(
                    pixel_values=batch.pixel_values.to(args.device)
                )
            outputs.append(F_normalize(value).cpu())
        return torch.cat(outputs)

    features: dict[str, torch.Tensor] = {}
    latencies: dict[str, list[float]] = {}
    for role, directory in roles.items():
        images, timings = _load_generated_images(directory, records)
        features[role] = encode(images)
        latencies[role] = timings
    features["real plant data"] = encode(_load_real_images(args.image_archive))
    torch.save({**expected, "features": features, "latencies": latencies}, cache_path)
    return features, latencies


def _set_metric_statistics(
    features: dict[str, torch.Tensor],
    latencies: dict[str, list[float]],
    records: list[tuple[str, int, str]],
    bandwidth: float,
) -> dict[str, dict[str, float]]:
    seed_positions: list[list[int]] = [[], [], [], []]
    by_prompt: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, (prompt_id, seed, _) in enumerate(records):
        by_prompt[prompt_id].append((seed, index))
    for values in by_prompt.values():
        for position, (_, index) in enumerate(sorted(values)):
            seed_positions[position].append(index)
    if len(by_prompt) != 30 or any(len(indices) != 30 for indices in seed_positions):
        raise ValueError("CMMD replicate partition is not 30 prompts x 4 seeds.")
    real = features["real plant data"]
    output: dict[str, dict[str, float]] = {}
    for role in ("20-step Teacher B", "Primary 4-step", "Primary 2-step"):
        cmmd_values = [
            unbiased_cmmd(features[role][indices], real, bandwidth)
            for indices in seed_positions
        ]
        latency_values = [
            statistics.median(latencies[role][index] for index in indices)
            for indices in seed_positions
        ]
        output[role] = {
            "seed_replicate_mean_cmmd": statistics.fmean(cmmd_values),
            "seed_replicate_sample_sd_cmmd": statistics.stdev(cmmd_values),
            "seed_replicate_sample_variance_cmmd": statistics.variance(cmmd_values),
            "seed_replicate_mean_median_seconds": statistics.fmean(latency_values),
            "seed_replicate_sample_sd_median_seconds": statistics.stdev(latency_values),
            "seed_replicate_sample_variance_median_seconds": statistics.variance(latency_values),
        }
    return output


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    four, two = _load(args.four_step_csv), _load(args.two_step_csv)
    if [(r["prompt_id"], r["seed"]) for r in four] != [
        (r["prompt_id"], r["seed"]) for r in two
    ]:
        raise ValueError("4-step and 2-step score files are not aligned.")
    teacher_a = _replicate_statistics(four, "teacher_clip")
    teacher_b = _replicate_statistics(two, "teacher_clip")
    if teacher_a != teacher_b:
        raise ValueError("Teacher scores differ between saved evaluations.")
    records = _image_records(four)
    features, latencies = _feature_cache(args, records)
    set_metrics = _set_metric_statistics(
        features, latencies, records, args.cmmd_bandwidth
    )
    models = {
        "20-step Teacher B": teacher_a,
        "Primary 4-step": _replicate_statistics(four, "student_clip"),
        "Primary 2-step": _replicate_statistics(two, "student_clip"),
    }
    for role in models:
        models[role].update(set_metrics[role])
    result = {
        "format_version": 1,
        "status": "PASS",
        "training_performed": False,
        "protocol": {
            "unseen_prompts": 30,
            "seeds_per_prompt": 4,
            "images_per_model": 120,
            "variance_definition": (
                "sample variance across four seed-replicate means; each replicate "
                "mean averages one fixed seed position across all 30 prompts"
            ),
            "cmmd_replicate_definition": (
                "one image per prompt for each seed position (30 generated images) "
                "compared with all 209 real plant images"
            ),
            "clip_image_feature_cache": str(
                args.output_dir.resolve() / "clip_image_feature_cache.pt"
            ),
        },
        "models": models,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "seed_variance_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    fields = (
        "model",
        "mean_clip_120_images",
        "seed_replicate_sample_sd",
        "seed_replicate_sample_variance",
        "rms_within_prompt_seed_sd",
        "seed_replicate_mean_cmmd",
        "seed_replicate_sample_sd_cmmd",
        "seed_replicate_sample_variance_cmmd",
        "seed_replicate_mean_median_seconds",
        "seed_replicate_sample_sd_median_seconds",
        "seed_replicate_sample_variance_median_seconds",
        "replicate_1_mean",
        "replicate_2_mean",
        "replicate_3_mean",
        "replicate_4_mean",
    )
    with (output_dir / "seed_variance_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model, stats in result["models"].items():
            writer.writerow({"model": model, **{key: stats[key] for key in fields[1:]}})
    print(json.dumps(result, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    summarize(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
