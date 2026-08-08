#!/usr/bin/env python3
"""Score a completed style-teacher checkpoint image evaluation with CLIP."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as functional
from PIL import Image


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_RANKS = (4, 8, 16, 32)


@dataclass(frozen=True)
class ImageItem:
    step: int
    label: str
    image_path: Path
    rank: int | None


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description=(
            "Calculate CLIP text-image cosine scores for the saved four-rank "
            "style-teacher checkpoint image evaluation and its official base image."
        )
    )
    parser.add_argument(
        "--source-evaluation-dir",
        type=Path,
        default=(
            root
            / "outputs"
            / "evaluation"
            / "style_teacher_checkpoint_grids_palm_adaptation"
        ),
        help="Existing image-evaluation directory whose metadata supplies the prompt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            root
            / "outputs"
            / "evaluation"
            / "style_teacher_clip_scores_palm_adaptation"
        ),
    )
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ranks", type=int, nargs="+", default=DEFAULT_RANKS)
    parser.add_argument("--expected-groups", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_source_metadata(source_dir: Path) -> dict[str, Any]:
    metadata = read_json(source_dir / "evaluation_metadata.json")
    if metadata.get("status") != "PASS":
        raise ValueError("Source evaluation metadata is not a PASS result.")
    prompt = metadata.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Source evaluation has no usable prompt.")
    return metadata


def discover_images(
    source_dir: Path,
    ranks: Sequence[int],
    expected_groups: int,
) -> list[ImageItem]:
    base_path = source_dir / "official_base_model.png"
    if not base_path.is_file():
        raise FileNotFoundError(f"Official base image not found: {base_path}")
    items = [ImageItem(0, "Official base", base_path, None)]
    rank_steps: dict[int, dict[int, Path]] = {}
    for rank in ranks:
        paths: dict[int, Path] = {}
        for image_path in (source_dir / "individual").glob(f"step_*_rank_{rank}.png"):
            parts = image_path.stem.split("_")
            if len(parts) != 4 or not parts[1].isdigit():
                continue
            step = int(parts[1])
            if step in paths:
                raise ValueError(f"Duplicate rank-{rank} image at step {step}.")
            paths[step] = image_path
        if not paths:
            raise FileNotFoundError(f"No rank-{rank} images found in {source_dir}")
        rank_steps[rank] = paths

    common_steps = set.intersection(*(set(values) for values in rank_steps.values()))
    all_steps = set.union(*(set(values) for values in rank_steps.values()))
    if common_steps != all_steps:
        missing = {
            str(rank): sorted(all_steps - set(values))
            for rank, values in rank_steps.items()
            if set(values) != all_steps
        }
        raise ValueError(f"Ranks have different image steps: {missing}")
    if len(common_steps) != expected_groups:
        raise ValueError(
            f"Expected {expected_groups} common steps, found {len(common_steps)}."
        )
    for step in sorted(common_steps):
        items.extend(
            ImageItem(step, f"Rank {rank}", rank_steps[rank][step], rank)
            for rank in ranks
        )
    return items


def calculate_scores(
    items: Sequence[ImageItem],
    prompt: str,
    clip_model: str,
    batch_size: int,
    device: str,
) -> list[float]:
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(clip_model, local_files_only=True)
    model = CLIPModel.from_pretrained(clip_model, local_files_only=True).to(device).eval()
    text_inputs = processor(
        text=[prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.inference_mode():
        text_outputs = model.text_model(
            input_ids=text_inputs["input_ids"].to(device),
            attention_mask=text_inputs["attention_mask"].to(device),
        )
        text_features = model.text_projection(text_outputs.pooler_output)
    text_features = functional.normalize(text_features.float(), dim=-1)
    scores: list[float] = []
    for offset in range(0, len(items), batch_size):
        batch = items[offset : offset + batch_size]
        images: list[Image.Image] = []
        try:
            for item in batch:
                with Image.open(item.image_path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            with torch.inference_mode():
                image_outputs = model.vision_model(
                    pixel_values=inputs["pixel_values"].to(device)
                )
                image_features = model.visual_projection(image_outputs.pooler_output)
            image_features = functional.normalize(image_features.float(), dim=-1)
            values = image_features @ text_features.T
            scores.extend(float(value) for value in values.squeeze(1).cpu())
        finally:
            for image in images:
                image.close()
    del model, processor
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return scores


def rows_from_scores(
    items: Sequence[ImageItem], scores: Sequence[float], ranks: Sequence[int]
) -> list[dict[str, Any]]:
    if len(items) != len(scores):
        raise ValueError("Image and score counts differ.")
    rows: dict[int, dict[str, Any]] = {}
    for item, score in zip(items, scores):
        row = rows.setdefault(item.step, {"step": item.step})
        if item.rank is None:
            row["official_base"] = score
        else:
            row[f"rank_{item.rank}"] = score
    expected_steps = sorted(rows)
    for step in expected_steps:
        row = rows[step]
        if step == 0:
            if "official_base" not in row:
                raise ValueError("Missing official base score.")
        elif any(f"rank_{rank}" not in row for rank in ranks):
            raise ValueError(f"Missing rank result at step {step}.")
    return [rows[step] for step in expected_steps]


def write_csv(path: Path, rows: Sequence[dict[str, Any]], ranks: Sequence[int]) -> None:
    fields = ["step", "official_base", *[f"rank_{rank}" for rank in ranks]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_table_figure(
    rows: Sequence[dict[str, Any]], ranks: Sequence[int], output: Path
) -> None:
    columns = ["Step", "Official base (Step 0)", *[f"Rank {rank}" for rank in ranks]]
    cells: list[list[str]] = []
    for row in rows:
        cells.append(
            [
                "0 (Base)" if row["step"] == 0 else f"{row['step']:,}",
                f"{row['official_base']:.4f}" if "official_base" in row else "-",
                *[
                    f"{row[f'rank_{rank}']:.4f}"
                    if f"rank_{rank}" in row
                    else "-"
                    for rank in ranks
                ],
            ]
        )
    figure, axis = plt.subplots(figsize=(13.2, 7.2))
    axis.axis("off")
    table = axis.table(
        cellText=cells,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.13, 0.24, 0.157, 0.157, 0.157, 0.157],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.75)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_facecolor("#edf2f7")
            cell.set_text_props(weight="bold")
        elif column == 0:
            cell.set_facecolor("#f8fafc")
            cell.set_text_props(weight="bold")
    axis.set_title(
        "CLIPScore table: official base vs. style-teacher checkpoints",
        fontsize=17,
        fontweight="bold",
        pad=20,
    )
    figure.text(
        0.5,
        0.03,
        "CLIP text-image cosine similarity; one identical prompt and seed per image.",
        ha="center",
        fontsize=10,
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_trajectory_figure(
    rows: Sequence[dict[str, Any]],
    ranks: Sequence[int],
    output: Path,
    baseline: float,
) -> None:
    checkpoint_rows = [row for row in rows if row["step"] > 0]
    steps = [row["step"] for row in checkpoint_rows]
    figure, axis = plt.subplots(figsize=(13.2, 7.2))
    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]
    for rank, color in zip(ranks, colors):
        values = [row[f"rank_{rank}"] for row in checkpoint_rows]
        axis.plot(
            steps,
            values,
            marker="o",
            linewidth=2.6,
            markersize=6,
            color=color,
            label=f"Rank {rank}",
        )
    axis.axhline(
        baseline,
        color="#7f8c8d",
        linestyle="--",
        linewidth=2.4,
        label=f"Official base (Step 0): {baseline:.4f}",
    )
    axis.set_title(
        "Style-teacher CLIPScore trajectory vs. official base model",
        fontsize=17,
        fontweight="bold",
    )
    axis.set_xlabel("Training steps")
    axis.set_ylabel("CLIPScore (text-image cosine similarity)")
    axis.set_xticks(steps)
    axis.grid(True, linestyle="--", alpha=0.45)
    axis.legend(title="Model", loc="best", frameon=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if sys.version_info[:2] != (3, 11):
        parser.error(f"Expected Python 3.11.x, got {platform.python_version()}.")
    if len(args.ranks) != 4 or len(set(args.ranks)) != 4:
        parser.error("--ranks must contain four distinct ranks.")
    if args.batch_size <= 0 or args.expected_groups <= 0:
        parser.error("--batch-size and --expected-groups must be positive.")
    if not args.dry_run and args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    source_dir = args.source_evaluation_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_metadata = load_source_metadata(source_dir)
    items = discover_images(source_dir, args.ranks, args.expected_groups)
    prompt = source_metadata["prompt"]
    print(f"Prompt: {prompt}")
    print(f"Scoring {len(items)} images: 1 official base + {len(items) - 1} LoRA outputs.")
    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    scores = calculate_scores(items, prompt, args.clip_model, args.batch_size, args.device)
    rows = rows_from_scores(items, scores, args.ranks)
    csv_path = output_dir / "clip_scores.csv"
    table_path = output_dir / "clip_score_table.png"
    trajectory_path = output_dir / "clip_score_trajectory.png"
    write_csv(csv_path, rows, args.ranks)
    make_table_figure(rows, args.ranks, table_path)
    baseline = rows[0]["official_base"]
    make_trajectory_figure(rows, args.ranks, trajectory_path, baseline)
    metadata = {
        "status": "PASS",
        "metric": "CLIP text-image cosine similarity",
        "clip_model": args.clip_model,
        "source_evaluation_dir": str(source_dir),
        "source_prompt": prompt,
        "source_prompt_sample_id": source_metadata.get("prompt_source_sample_id"),
        "seed": source_metadata.get("seed"),
        "num_inference_steps": source_metadata.get("num_inference_steps"),
        "guidance_scale": source_metadata.get("guidance_scale"),
        "official_base_image": str(items[0].image_path),
        "scored_images": [
            {
                "step": item.step,
                "rank": item.rank,
                "label": item.label,
                "image": str(item.image_path),
                "clip_score": score,
            }
            for item, score in zip(items, scores)
        ],
        "outputs": {
            "csv": str(csv_path),
            "table": str(table_path),
            "trajectory": str(trajectory_path),
        },
    }
    (output_dir / "clip_score_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"PASS: wrote {table_path}, {trajectory_path}, and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

