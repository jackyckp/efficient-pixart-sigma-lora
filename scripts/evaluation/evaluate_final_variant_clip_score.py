#!/usr/bin/env python3
"""Compare final LoRA variants from one saved prompt-generation directory."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.evaluate_style_teacher_clip_score import (  # noqa: E402
    ImageItem,
    calculate_scores,
)


BASE_NAME = "official_base"
LORA_PATTERN = re.compile(r"r(?P<rank>\d+)_lr(?P<learning_rate>.+)$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate CLIPScores for final LoRA variants and the official base."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def display_label(name: str) -> str:
    if name == BASE_NAME:
        return "Official base\n(Step 0)"
    match = LORA_PATTERN.fullmatch(name)
    if not match:
        return name
    learning_rate = match.group("learning_rate").replace("e-0", "e-")
    return f"Rank {match.group('rank')}\nLR {learning_rate}"


def discover_variants(source_dir: Path) -> tuple[str, list[ImageItem], list[dict[str, Any]]]:
    base_json = source_dir / f"{BASE_NAME}.json"
    if not base_json.is_file():
        raise FileNotFoundError(f"Missing base metadata: {base_json}")
    metadata_files = [base_json, *sorted(source_dir.glob("r*_lr*.json"))]
    items: list[ImageItem] = []
    metadata: list[dict[str, Any]] = []
    prompt: str | None = None
    for index, json_path in enumerate(metadata_files):
        payload = read_json(json_path)
        if payload.get("status") != "PASS":
            raise ValueError(f"Generation did not pass: {json_path}")
        current_prompt = payload.get("prompt")
        image_value = payload.get("image")
        if not isinstance(current_prompt, str) or not current_prompt.strip():
            raise ValueError(f"Missing prompt: {json_path}")
        if not isinstance(image_value, str):
            raise ValueError(f"Missing image path: {json_path}")
        image_path = Path(image_value)
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing generated image: {image_path}")
        if prompt is None:
            prompt = current_prompt
        elif prompt != current_prompt:
            raise ValueError("Variant files use different prompts.")
        name = json_path.stem
        items.append(ImageItem(index, display_label(name), image_path, None))
        metadata.append(
            {
                "name": name,
                "label": display_label(name).replace("\n", " | "),
                "generation_metadata": payload,
            }
        )
    assert prompt is not None
    if len(items) != 5:
        raise ValueError(f"Expected base plus four LoRA variants, found {len(items)}.")
    return prompt, items, metadata


def write_table(
    labels: Sequence[str], scores: Sequence[float], output: Path
) -> None:
    figure, axis = plt.subplots(figsize=(13.2, 3.7))
    axis.axis("off")
    table = axis.table(
        cellText=[[f"{score:.4f}" for score in scores]],
        colLabels=labels,
        rowLabels=["CLIPScore"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.2)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_facecolor("#edf2f7")
            cell.set_text_props(weight="bold")
        elif row == 1:
            cell.set_facecolor("#ffffff")
    axis.set_title(
        "Plant-only style teacher: final-variant CLIPScore comparison",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    figure.text(
        0.5,
        0.06,
        "CLIP text-image cosine similarity; identical prompt, seed, 20 steps, and CFG 1.0.",
        ha="center",
        fontsize=10,
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_bar_chart(
    labels: Sequence[str], scores: Sequence[float], output: Path
) -> None:
    colors = ["#7f8c8d", "#e74c3c", "#e67e22", "#2ecc71", "#3498db"]
    figure, axis = plt.subplots(figsize=(11.8, 6.2))
    bars = axis.bar(labels, scores, color=colors, width=0.68)
    baseline = scores[0]
    axis.axhline(
        baseline,
        color="#7f8c8d",
        linestyle="--",
        linewidth=2,
        label=f"Official base: {baseline:.4f}",
    )
    for bar, score in zip(bars, scores):
        axis.annotate(
            f"{score:.4f}",
            (bar.get_x() + bar.get_width() / 2, score),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    axis.set_title(
        "Plant-only final-model CLIPScore comparison",
        fontsize=16,
        fontweight="bold",
    )
    axis.set_ylabel("CLIPScore (text-image cosine similarity)")
    axis.grid(axis="y", linestyle="--", alpha=0.45)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    source_dir, output_dir = args.source_dir.resolve(), args.output_dir.resolve()
    prompt, items, variant_metadata = discover_variants(source_dir)
    print(f"Scoring {len(items)} variants with prompt: {prompt}")
    scores = calculate_scores(
        items, prompt, args.clip_model, args.batch_size, args.device
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [item.label for item in items]
    rows = [
        {
            "variant": metadata["name"],
            "label": metadata["label"],
            "clip_score": score,
            "image": str(item.image_path),
        }
        for item, metadata, score in zip(items, variant_metadata, scores)
    ]
    csv_path = output_dir / "clip_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    table_path = output_dir / "clip_score_table.png"
    bar_path = output_dir / "clip_score_bar_chart.png"
    write_table(labels, scores, table_path)
    write_bar_chart(labels, scores, bar_path)
    output_metadata = {
        "status": "PASS",
        "metric": "CLIP text-image cosine similarity",
        "clip_model": args.clip_model,
        "prompt": prompt,
        "source_dir": str(source_dir),
        "variants": rows,
        "outputs": {
            "csv": str(csv_path),
            "table": str(table_path),
            "bar_chart": str(bar_path),
        },
    }
    (output_dir / "clip_score_metadata.json").write_text(
        json.dumps(output_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"PASS: wrote {table_path}, {bar_path}, and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

