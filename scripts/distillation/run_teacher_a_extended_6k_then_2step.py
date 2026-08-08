#!/usr/bin/env python3
"""Continue Teacher A's accepted 4k student to 6k, then train a fresh 2-step."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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


TEACHER_ID = "plant_n209_r16_step10200"


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description=(
            "Resume-safe isolated experiment: Teacher A 4-step 4k->6k, "
            "then a fresh 7k 2-step student."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            root
            / "outputs"
            / "distillation_experiments"
            / "teacher_a_extend6k_then2step"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def execute(command: Sequence[str], *, dry_run: bool) -> None:
    print("\nCOMMAND")
    print(subprocess.list2cmdline([str(value) for value in command]), flush=True)
    if not dry_run:
        subprocess.run([str(value) for value in command], check=True)


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
        raise FileNotFoundError("Missing 4k experiment inputs:\n" + "\n".join(missing))
    checkpoint = read_json(source_checkpoint / "checkpoint_metadata.json")
    if not (
        checkpoint.get("teacher_id") == TEACHER_ID
        and checkpoint.get("target_inference_steps") == 4
        and checkpoint.get("optimizer_step") == 4_000
        and checkpoint.get("max_train_steps") == 4_000
        and checkpoint.get("status") == "CHECKPOINT"
    ):
        raise ValueError("Teacher A extended step_004000 checkpoint mismatch.")
    evaluation = read_json(source_evaluation)
    if not (
        evaluation.get("status") == "PASS"
        and evaluation.get("student_steps") == 4
        and evaluation.get("image_count_per_model") == 120
    ):
        raise ValueError("The source 4k adapter has not passed formal evaluation.")
    return checkpoint


def seed_previous_best(
    *,
    output_dir: Path,
    source_adapter: Path,
    source_checkpoint: Path,
) -> None:
    """Keep step 4000 eligible when selecting the best 4k-6k adapter."""
    best_adapter = output_dir / "best_adapter"
    selection = output_dir / "best_checkpoint.json"
    if best_adapter.exists() or selection.exists():
        if not (
            best_adapter.is_dir()
            and (best_adapter / "adapter_model.safetensors").is_file()
            and selection.is_file()
        ):
            raise RuntimeError(
                f"Partially initialized best-adapter seed in {output_dir}."
            )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_adapter, best_adapter)
    write_json(
        selection,
        {
            "optimizer_step": 4_000,
            "checkpoint": str(source_checkpoint),
            "selection_metric": "mean checkpoint interval training loss",
            "note": "Seeded from the formally evaluated 4k experiment.",
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = repository_root()
    source_root = (
        root
        / "outputs"
        / "distillation_experiments"
        / "teacher_a_extend4k_then2step"
    ).resolve()
    protected_roots = (
        (root / "outputs" / "distillation" / TEACHER_ID).resolve(),
        source_root,
    )
    output_root = args.output_root.resolve()
    if any(
        output_root == protected or protected in output_root.parents
        for protected in protected_roots
    ):
        raise ValueError(
            "The 6k experiment output must be outside all accepted source runs."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    extended_six = output_root / "student_4step_extended_to6000"
    new_two = output_root / "student_2step_from_6000_4step"
    source_four = source_root / "student_4step_extended_to4000"
    source_adapter = source_four / "best_adapter"
    source_checkpoint = source_four / "checkpoints" / "step_004000"
    source_evaluation = (
        source_root
        / "evaluation_4step"
        / "metrics"
        / "evaluation_summary.json"
    )
    baseline = root / "outputs" / "distillation" / TEACHER_ID
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
        print(f"AUTO-RESUME 4-step extension from {resume}", flush=True)
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
        raise FileNotFoundError(f"Extended 6k best adapter missing: {extended_best}")

    if not completed_run(new_two, target_steps=2, optimizer_steps=7_000):
        resume = latest_checkpoint(
            new_two, target_steps=2, max_train_steps=7_000
        )
        if resume is not None:
            print(f"AUTO-RESUME new 2-step training from {resume}", flush=True)
        execute(
            training_command(
                trajectory_cache=trajectory_cache,
                prompt_cache=prompt_cache,
                init_adapter=extended_best,
                output_dir=new_two,
                target_steps=2,
                max_train_steps=7_000,
                checkpoint_every=1_000,
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

