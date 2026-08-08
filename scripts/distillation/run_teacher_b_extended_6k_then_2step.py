#!/usr/bin/env python3
"""Continue Teacher B's accepted 4k student to 6k, then train a fresh 2-step."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.distillation.common import repository_root, write_json  # noqa: E402
from scripts.distillation.run_teacher_a_extended_4step_then_2step import (  # noqa: E402
    completed_run,
    latest_checkpoint,
    read_json,
    training_command,
)
from scripts.distillation.run_teacher_a_extended_6k_then_2step import (  # noqa: E402
    execute,
    seed_previous_best,
)


TEACHER_ID = "teammate_plant209_step4000"


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description=(
            "Resume-safe isolated experiment: Teacher B 4-step 4k->6k, "
            "then a fresh 7k 2-step student with 2k checkpoint intervals."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            root
            / "outputs"
            / "distillation_experiments"
            / "teacher_b_extend6k_then2step"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_source_assets(
    *,
    trajectory_cache: Path,
    prompt_cache: Path,
    source_adapter: Path,
    source_checkpoint: Path,
    source_evaluation: Path,
) -> dict[str, Any]:
    required = (
        trajectory_cache / "cache_manifest.json",
        prompt_cache,
        source_adapter / "adapter_config.json",
        source_adapter / "adapter_model.safetensors",
        source_checkpoint / "checkpoint_metadata.json",
        source_checkpoint / "training_state.pt",
        source_checkpoint / "lora_adapter" / "adapter_model.safetensors",
        source_evaluation,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing accepted Teacher B 4k inputs:\n" + "\n".join(missing)
        )
    checkpoint = read_json(source_checkpoint / "checkpoint_metadata.json")
    if not (
        checkpoint.get("teacher_id") == TEACHER_ID
        and checkpoint.get("target_inference_steps") == 4
        and checkpoint.get("optimizer_step") == 4_000
        and checkpoint.get("max_train_steps") == 4_000
        and checkpoint.get("status") == "CHECKPOINT"
    ):
        raise ValueError("Teacher B extended step_004000 checkpoint mismatch.")
    evaluation = read_json(source_evaluation)
    if not (
        evaluation.get("status") == "PASS"
        and evaluation.get("student_steps") == 4
        and evaluation.get("image_count_per_model") == 120
        and all(evaluation.get("gates", {}).values())
    ):
        raise ValueError("Teacher B source 4k model has not passed every formal gate.")
    return checkpoint


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = repository_root()
    baseline = (root / "outputs" / "distillation" / TEACHER_ID).resolve()
    source_root = (
        root
        / "outputs"
        / "distillation_experiments"
        / "teacher_b_extend4k_then2step"
    ).resolve()
    output_root = args.output_root.resolve()
    protected_roots = (baseline, source_root)
    if any(
        output_root == protected or protected in output_root.parents
        for protected in protected_roots
    ):
        raise ValueError(
            "The Teacher B 6k experiment must be outside all accepted source runs."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    extended_six = output_root / "student_4step_extended_to6000"
    new_two = output_root / "student_2step_from_6000_4step"
    source_four = source_root / "student_4step_extended_to4000"
    source_adapter = source_four / "best_adapter"
    source_checkpoint = source_four / "checkpoints" / "step_004000"
    source_evaluation = (
        source_root / "evaluation_4step" / "metrics" / "evaluation_summary.json"
    )
    trajectory_cache = baseline / "trajectory_cache_v1"
    prompt_cache = (
        root / "data" / "features" / "distill_t5_plant627_len300_fp16_v1.pt"
    )
    if not args.dry_run:
        validate_source_assets(
            trajectory_cache=trajectory_cache,
            prompt_cache=prompt_cache,
            source_adapter=source_adapter,
            source_checkpoint=source_checkpoint,
            source_evaluation=source_evaluation,
        )

    summary_path = output_root / "experiment_summary.json"
    state: dict[str, Any] = {
        "format_version": 1,
        "status": "RUNNING",
        "teacher_id": TEACHER_ID,
        "source_4step_checkpoint": str(source_checkpoint),
        "source_4step_evaluation": str(source_evaluation),
        "extended_6k_output": str(extended_six),
        "new_2step_output": str(new_two),
        "updated_unix_time": time.time(),
    }
    write_json(summary_path, state)

    if not completed_run(extended_six, target_steps=4, optimizer_steps=6_000):
        if not args.dry_run:
            seed_previous_best(
                output_dir=extended_six,
                source_adapter=source_adapter,
                source_checkpoint=source_checkpoint,
            )
        resume = latest_checkpoint(
            extended_six, target_steps=4, max_train_steps=6_000
        )
        if resume is None:
            resume = source_checkpoint
        print(f"AUTO-RESUME Teacher B 4-step extension from {resume}", flush=True)
        execute(
            training_command(
                trajectory_cache=trajectory_cache,
                prompt_cache=prompt_cache,
                init_adapter=source_adapter,
                output_dir=extended_six,
                target_steps=4,
                max_train_steps=6_000,
                checkpoint_every=500,
                learning_rate=5e-6,
                resume_from=resume,
            ),
            dry_run=args.dry_run,
        )
    extended_best = extended_six / "best_adapter"
    if not args.dry_run and not (
        extended_best / "adapter_model.safetensors"
    ).is_file():
        raise FileNotFoundError(f"Teacher B extended 6k best missing: {extended_best}")

    if not completed_run(new_two, target_steps=2, optimizer_steps=7_000):
        resume = latest_checkpoint(new_two, target_steps=2, max_train_steps=7_000)
        if resume is not None:
            print(f"AUTO-RESUME Teacher B fresh 2-step from {resume}", flush=True)
        execute(
            training_command(
                trajectory_cache=trajectory_cache,
                prompt_cache=prompt_cache,
                init_adapter=extended_best,
                output_dir=new_two,
                target_steps=2,
                max_train_steps=7_000,
                checkpoint_every=2_000,
                learning_rate=2e-6,
                resume_from=resume,
            ),
            dry_run=args.dry_run,
        )
    state.update(
        {
            "status": "PLANNED" if args.dry_run else "PASS",
            "extended_6k_best_adapter": str(extended_best),
            "new_2step_best_adapter": str(new_two / "best_adapter"),
            "updated_unix_time": time.time(),
        }
    )
    write_json(summary_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return state


def main(argv: Sequence[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
