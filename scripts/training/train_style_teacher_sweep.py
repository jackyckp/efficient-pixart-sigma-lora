#!/usr/bin/env python3
"""Run the four-rank PixArt 20-step style-teacher sweep sequentially."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


DEFAULT_RANKS = (4, 8, 16, 32)
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_MAX_TRAIN_STEPS = 10_000
DEFAULT_CHECKPOINT_EVERY_STEPS = 1_000
FIXED_INFERENCE_STEPS = 20
FIXED_GUIDANCE_SCALE = 1.5
CANONICAL_DATASET_COUNT = 260
REFERENCE_SECONDS_PER_STEP = 0.702943227


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def learning_rate_slug(value: float) -> str:
    return f"{value:.0e}".replace("+", "")


def build_run_matrix(
    ranks: Sequence[int],
    learning_rate: float,
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "learning_rate": learning_rate,
            "name": f"r{rank}_lr{learning_rate_slug(learning_rate)}",
        }
        for rank in ranks
    ]


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description=(
            "Validate all four canonical datasets, then train four LoRA "
            "ranks sequentially at one learning rate."
        )
    )
    parser.add_argument("--ranks", type=int, nargs="+", default=DEFAULT_RANKS)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=DEFAULT_MAX_TRAIN_STEPS,
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_STEPS,
    )
    parser.add_argument("--num-images", type=int, default=CANONICAL_DATASET_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
    )
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            root / "outputs" / "style_teacher" / "all_n260_steps10000"
        ),
    )
    parser.add_argument(
        "--trainer",
        type=Path,
        default=root / "scripts" / "training" / "train_local_latent_lora.py",
    )
    parser.add_argument(
        "--latent-bundle",
        type=Path,
        default=root / "data" / "archives" / "clean_latents_512.zip",
    )
    parser.add_argument(
        "--image-archive",
        type=Path,
        default=root / "data" / "archives" / "ink.zip",
    )
    parser.add_argument(
        "--prompt-cache",
        type=Path,
        default=(
            root
            / "data"
            / "features"
            / "t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt"
        ),
    )
    parser.add_argument(
        "--prompt-validation-summary",
        type=Path,
        default=root / "data" / "features" / "validation_summary.json",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Train again even when matching PASS metadata already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print/write the four commands only.",
    )
    return parser


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if sys.version_info[:3] != (3, 11, 2):
        parser.error(
            f"Expected Python 3.11.2, got {platform.python_version()}."
        )
    if len(args.ranks) != 4 or len(set(args.ranks)) != 4:
        parser.error("--ranks must contain exactly four distinct values.")
    if any(rank <= 0 for rank in args.ranks):
        parser.error("--ranks must be positive.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive.")
    for name in (
        "max_train_steps",
        "checkpoint_every_steps",
        "num_images",
        "train_batch_size",
        "gradient_accumulation_steps",
        "log_every_steps",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.num_images != CANONICAL_DATASET_COUNT:
        parser.error(
            f"This sweep requires all {CANONICAL_DATASET_COUNT} images "
            "from animal, others, plant, and web."
        )
    if args.checkpoint_every_steps > args.max_train_steps:
        parser.error("--checkpoint-every-steps exceeds training length.")
    if args.max_train_steps % args.checkpoint_every_steps != 0:
        parser.error(
            "--max-train-steps must be divisible by "
            "--checkpoint-every-steps."
        )


def common_trainer_args(args: argparse.Namespace) -> list[str]:
    return [
        str(args.trainer.resolve()),
        "--latent-bundle",
        str(args.latent_bundle.resolve()),
        "--image-archive",
        str(args.image_archive.resolve()),
        "--prompt-cache",
        str(args.prompt_cache.resolve()),
        "--prompt-validation-summary",
        str(args.prompt_validation_summary.resolve()),
        "--num-images",
        str(args.num_images),
        "--max-train-steps",
        str(args.max_train_steps),
        "--checkpoint-every-steps",
        str(args.checkpoint_every_steps),
        "--train-batch-size",
        str(args.train_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--seed",
        str(args.seed),
        "--inference-steps",
        str(FIXED_INFERENCE_STEPS),
        "--guidance-scale",
        str(FIXED_GUIDANCE_SCALE),
        "--log-every-steps",
        str(args.log_every_steps),
        "--run-role",
        "style_teacher",
    ]


def metadata_matches(
    metadata_path: Path,
    run: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(
        (
            metadata.get("status") == "PASS",
            metadata.get("run_role") == "style_teacher",
            metadata.get("dataset_category") is None,
            metadata.get("num_images") == args.num_images,
            metadata.get("rank") == run["rank"],
            metadata.get("learning_rate") == run["learning_rate"],
            metadata.get("optimizer_steps") == args.max_train_steps,
            metadata.get("checkpoint_every_steps")
            == args.checkpoint_every_steps,
            metadata.get("inference_steps") == FIXED_INFERENCE_STEPS,
            metadata.get("guidance_scale") == FIXED_GUIDANCE_SCALE,
        )
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    matrix = build_run_matrix(args.ranks, args.learning_rate)
    common = common_trainer_args(args)
    estimated_hours = (
        args.max_train_steps * REFERENCE_SECONDS_PER_STEP / 3600
    )
    plan: dict[str, Any] = {
        "status": "PLANNED",
        "python": platform.python_version(),
        "dataset_scope": "all_categories",
        "category_counts": {
            "animal": 30,
            "others": 10,
            "plant": 209,
            "web": 11,
        },
        "num_images": args.num_images,
        "max_train_steps_per_model": args.max_train_steps,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "learning_rate": args.learning_rate,
        "fixed_inference_steps": FIXED_INFERENCE_STEPS,
        "fixed_guidance_scale": FIXED_GUIDANCE_SCALE,
        "estimated_seconds_per_step": REFERENCE_SECONDS_PER_STEP,
        "estimated_hours_per_model": estimated_hours,
        "estimated_total_hours": estimated_hours * len(matrix),
        "runs": [],
    }
    for run in matrix:
        run_dir = output_root / run["name"]
        command = [
            sys.executable,
            *common,
            "--rank",
            str(run["rank"]),
            "--learning-rate",
            str(run["learning_rate"]),
            "--output-dir",
            str(run_dir),
        ]
        plan["runs"].append(
            {
                **run,
                "output_dir": str(run_dir),
                "command": command,
            }
        )
    write_json(output_root / "sweep_plan.json", plan)

    print(
        f"Teacher sweep: {len(matrix)} sequential models; "
        f"images={args.num_images} across all four categories; "
        f"updates/model={args.max_train_steps}; "
        f"inference={FIXED_INFERENCE_STEPS} steps; "
        f"checkpoint every {args.checkpoint_every_steps}; "
        f"guidance={FIXED_GUIDANCE_SCALE}; "
        f"estimated total={plan['estimated_total_hours']:.2f}h."
    )
    for item in plan["runs"]:
        print(subprocess.list2cmdline(item["command"]))
    if args.dry_run:
        print(f"DRY RUN: wrote {output_root / 'sweep_plan.json'}")
        return 0

    validation_command = [
        sys.executable,
        *common,
        "--rank",
        str(args.ranks[0]),
        "--learning-rate",
        str(args.learning_rate),
        "--validate-assets-only",
    ]
    subprocess.run(validation_command, check=True)

    results: list[dict[str, Any]] = []
    sweep_start = time.perf_counter()
    try:
        for item in plan["runs"]:
            run_dir = Path(item["output_dir"])
            metadata_path = run_dir / "run_metadata.json"
            if (
                not args.rerun_completed
                and metadata_matches(metadata_path, item, args)
            ):
                print(f"SKIP completed: {item['name']}")
                results.append(
                    {
                        "name": item["name"],
                        "status": "SKIPPED_COMPLETED",
                        "output_dir": item["output_dir"],
                    }
                )
                continue
            print(f"START: {item['name']}")
            run_start = time.perf_counter()
            subprocess.run(item["command"], check=True)
            elapsed = time.perf_counter() - run_start
            if not metadata_matches(metadata_path, item, args):
                raise RuntimeError(
                    f"Completed run metadata does not match: {metadata_path}"
                )
            results.append(
                {
                    "name": item["name"],
                    "status": "PASS",
                    "elapsed_seconds": elapsed,
                    "output_dir": item["output_dir"],
                }
            )
            write_json(
                output_root / "sweep_summary.json",
                {
                    "status": "RUNNING",
                    "results": results,
                    "elapsed_seconds": time.perf_counter() - sweep_start,
                },
            )
    except BaseException:
        write_json(
            output_root / "sweep_summary.json",
            {
                "status": "FAILED",
                "results": results,
                "elapsed_seconds": time.perf_counter() - sweep_start,
            },
        )
        raise

    write_json(
        output_root / "sweep_summary.json",
        {
            "status": "PASS",
            "results": results,
            "elapsed_seconds": time.perf_counter() - sweep_start,
        },
    )
    print(f"PASS: all teacher runs complete under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
