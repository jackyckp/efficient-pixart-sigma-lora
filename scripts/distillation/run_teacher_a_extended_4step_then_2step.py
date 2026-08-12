#!/usr/bin/env python3
"""Continue Teacher A's 4-step student to 4k, then train a fresh 7k 2-step."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.distillation.common import repository_root, write_json  # noqa: E402


STEP_PATTERN = re.compile(r"^step_(\d{6})$")
TEACHER_ID = "plant_n209_r16_step10200"


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description=(
            "Resume-safe isolated experiment: Teacher A 4-step 2k->4k, "
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
            / "teacher_a_extend4k_then2step"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def completed_run(
    output_dir: Path,
    *,
    target_steps: int,
    optimizer_steps: int,
) -> bool:
    path = output_dir / "run_metadata.json"
    if not path.is_file():
        return False
    try:
        metadata = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        metadata.get("status") == "PASS"
        and metadata.get("target_inference_steps") == target_steps
        and metadata.get("optimizer_steps") == optimizer_steps
        and metadata.get("max_train_steps") == optimizer_steps
    )


def latest_checkpoint(
    output_dir: Path,
    *,
    target_steps: int,
    max_train_steps: int,
) -> Path | None:
    """Find the newest complete checkpoint strictly before the terminal step."""
    root = output_dir / "checkpoints"
    if not root.is_dir():
        return None
    valid: list[tuple[int, Path]] = []
    for candidate in root.iterdir():
        match = STEP_PATTERN.fullmatch(candidate.name)
        if match is None or not candidate.is_dir():
            continue
        step = int(match.group(1))
        if not 0 < step < max_train_steps:
            continue
        metadata_path = candidate / "checkpoint_metadata.json"
        required = (
            metadata_path,
            candidate / "training_state.pt",
            candidate / "lora_adapter" / "adapter_config.json",
            candidate / "lora_adapter" / "adapter_model.safetensors",
        )
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            continue
        try:
            metadata = read_json(metadata_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        expected = {
            "status": "CHECKPOINT",
            "target_inference_steps": target_steps,
            "max_train_steps": max_train_steps,
            "optimizer_step": step,
            "rank": 16,
            "lora_alpha": 16,
        }
        if all(metadata.get(key) == value for key, value in expected.items()):
            valid.append((step, candidate.resolve()))
    return max(valid, default=(0, None), key=lambda item: item[0])[1]


def training_command(
    *,
    trajectory_cache: Path,
    prompt_cache: Path,
    init_adapter: Path,
    output_dir: Path,
    target_steps: int,
    max_train_steps: int,
    checkpoint_every: int,
    learning_rate: float,
    resume_from: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(
            repository_root()
            / "scripts"
            / "distillation"
            / "train_phased_distill_lora.py"
        ),
        "--trajectory-cache",
        str(trajectory_cache),
        "--prompt-cache",
        str(prompt_cache),
        "--init-adapter",
        str(init_adapter),
        "--target-steps",
        str(target_steps),
        "--max-train-steps",
        str(max_train_steps),
        "--learning-rate",
        str(learning_rate),
        "--checkpoint-every-steps",
        str(checkpoint_every),
        "--output-dir",
        str(output_dir),
    ]
    if resume_from is not None:
        command.extend(("--resume-from", str(resume_from)))
    return command


def execute(command: Sequence[str], *, dry_run: bool) -> None:
    print("\nCOMMAND")
    print(subprocess.list2cmdline([str(value) for value in command]), flush=True)
    if not dry_run:
        subprocess.run([str(value) for value in command], check=True)


def validate_source_assets(
    *,
    trajectory_cache: Path,
    prompt_cache: Path,
    baseline_adapter: Path,
    baseline_checkpoint: Path,
) -> None:
    required = (
        trajectory_cache / "cache_manifest.json",
        prompt_cache,
        baseline_adapter / "adapter_config.json",
        baseline_adapter / "adapter_model.safetensors",
        baseline_checkpoint / "checkpoint_metadata.json",
        baseline_checkpoint / "training_state.pt",
        baseline_checkpoint / "lora_adapter" / "adapter_model.safetensors",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing experiment inputs:\n" + "\n".join(missing))
    metadata = read_json(baseline_checkpoint / "checkpoint_metadata.json")
    if not (
        metadata.get("teacher_id") == TEACHER_ID
        and metadata.get("target_inference_steps") == 4
        and metadata.get("optimizer_step") == 2_000
        and metadata.get("status") == "CHECKPOINT"
    ):
        raise ValueError("Teacher A baseline step_002000 checkpoint mismatch.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = repository_root()
    protected = (
        root / "outputs" / "distillation" / TEACHER_ID
    ).resolve()
    output_root = args.output_root.resolve()
    if output_root == protected or protected in output_root.parents:
        raise ValueError(
            "Experiment output must not be inside the accepted baseline directory."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "experiment_summary.json"
    extended_four = output_root / "student_4step_extended_to4000"
    new_two = output_root / "student_2step_from_extended_4step"
    baseline = root / "outputs" / "distillation" / TEACHER_ID
    baseline_adapter = baseline / "student_4step" / "best_adapter"
    baseline_checkpoint = (
        baseline / "student_4step" / "checkpoints" / "step_002000"
    )
    trajectory_cache = baseline / "trajectory_cache_v1"
    prompt_cache = (
        root / "data" / "features" / "distill_t5_plant627_len300_fp16_v1.pt"
    )
    if not args.dry_run:
        validate_source_assets(
            trajectory_cache=trajectory_cache,
            prompt_cache=prompt_cache,
            baseline_adapter=baseline_adapter,
            baseline_checkpoint=baseline_checkpoint,
        )
    state: dict[str, Any] = {
        "format_version": 1,
        "status": "RUNNING",
        "teacher_id": TEACHER_ID,
        "baseline_4step_checkpoint": str(baseline_checkpoint),
        "extended_4step_output": str(extended_four),
        "new_2step_output": str(new_two),
        "updated_unix_time": time.time(),
    }
    write_json(summary_path, state)

    if not completed_run(extended_four, target_steps=4, optimizer_steps=4_000):
        resume = latest_checkpoint(
            extended_four, target_steps=4, max_train_steps=4_000
        )
        if resume is None:
            resume = baseline_checkpoint
        print(f"AUTO-RESUME 4-step extension from {resume}", flush=True)
        execute(
            training_command(
                trajectory_cache=trajectory_cache,
                prompt_cache=prompt_cache,
                init_adapter=baseline_adapter,
                output_dir=extended_four,
                target_steps=4,
                max_train_steps=4_000,
                checkpoint_every=500,
                learning_rate=5e-6,
                resume_from=resume,
            ),
            dry_run=args.dry_run,
        )
    extended_best = extended_four / "best_adapter"
    if not args.dry_run and not (
        extended_best / "adapter_model.safetensors"
    ).is_file():
        raise FileNotFoundError(f"Extended 4-step best adapter missing: {extended_best}")

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
            "extended_4step_best_adapter": str(extended_best),
            "new_2step_best_adapter": str(new_two / "best_adapter"),
            "updated_unix_time": time.time(),
        }
    )
    write_json(summary_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return state


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

