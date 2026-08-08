#!/usr/bin/env python3
"""Run the complete two-teacher 20 -> 4 -> 2 distillation workflow."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.distillation.common import (  # noqa: E402
    load_distill_prompt_cache,
    repository_root,
    write_json,
)


TEACHERS = (
    {
        "teacher_id": "plant_n209_r16_step10200",
        "adapter": "outputs/style_teacher/plant_n209_steps10200/r16_lr1e-05",
        "guidance_scale": 1.0,
    },
    {
        "teacher_id": "teammate_plant209_step4000",
        "adapter": "outputs/style_teacher/best_ink_wash_lora_plant209_step4000",
        "guidance_scale": 1.5,
    },
)


class QualityGateError(RuntimeError):
    """Raised when a 4-step student may not safely initialize 2-step training."""


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description=(
            "Resume-safe end-to-end two-teacher distillation: validate, cache, "
            "train/evaluate 4-step, train/evaluate 2-step."
        )
    )
    parser.add_argument(
        "--prompt-cache",
        type=Path,
        default=(
            root
            / "data"
            / "features"
            / "distill_t5_plant627_len300_fp16_v1.pt"
        ),
    )
    parser.add_argument(
        "--evaluation-prompts",
        type=Path,
        default=root / "evaluation" / "distillation_prompts_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "outputs" / "distillation",
    )
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument(
        "--skip-final-2step-evaluation",
        action="store_true",
        help="Finish after both 2-step training runs; not suitable for reporting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every command without loading assets or starting GPU work.",
    )
    return parser


def script_path(name: str) -> str:
    return str(repository_root() / "scripts" / "distillation" / name)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def matching_pass(path: Path, **expected: Any) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return payload.get("status") == "PASS" and all(
        payload.get(key) == value for key, value in expected.items()
    )


def validate_prompt_assets(
    prompt_cache: Path,
    evaluation_prompts: Path,
) -> dict[str, Any]:
    if not prompt_cache.is_file():
        raise FileNotFoundError(
            "The 627-prompt T5 cache is not ready. Let "
            "build_distill_prompt_cache.py finish before starting this runner: "
            f"{prompt_cache}"
        )
    if not evaluation_prompts.is_file():
        raise FileNotFoundError(
            f"Evaluation prompt manifest is not ready: {evaluation_prompts}"
        )
    features = load_distill_prompt_cache(prompt_cache)
    if len(features.prompt_ids) != 627:
        raise ValueError(
            f"Formal training requires 627 prompts, got {len(features.prompt_ids)}."
        )
    if len(set(features.source_sample_ids)) != 209:
        raise ValueError("Formal prompt cache must cover exactly 209 plant IDs.")
    evaluation = read_json(evaluation_prompts)
    prompts = evaluation.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 30:
        raise ValueError("Formal evaluation requires exactly 30 held-out prompts.")
    if any(len(record.get("seeds", [])) != 4 for record in prompts):
        raise ValueError("Every held-out prompt must define exactly four seeds.")
    if evaluation.get("prompt_bank_fingerprint") != (
        features.prompt_bank_fingerprint
    ):
        raise ValueError("Evaluation/prompt-cache fingerprint mismatch.")
    result = {
        "prompt_cache": str(prompt_cache),
        "prompt_count": len(features.prompt_ids),
        "source_sample_count": len(set(features.source_sample_ids)),
        "evaluation_prompt_count": len(prompts),
        "evaluation_seeds_per_prompt": 4,
        "prompt_bank_fingerprint": features.prompt_bank_fingerprint,
    }
    del features
    gc.collect()
    return result


def run_command(
    command: Sequence[str],
    *,
    dry_run: bool,
    command_log: list[list[str]],
) -> None:
    normalized = [str(value) for value in command]
    command_log.append(normalized)
    print("\nCOMMAND")
    print(subprocess.list2cmdline(normalized), flush=True)
    if not dry_run:
        subprocess.run(normalized, check=True)


def update_state(
    path: Path,
    state: dict[str, Any],
    stage: str,
    status: str = "RUNNING",
) -> None:
    state["stage"] = stage
    state["status"] = status
    state["updated_unix_time"] = time.time()
    write_json(path, state)


def validation_command(
    teacher: dict[str, Any], manifest: Path
) -> list[str]:
    return [
        sys.executable,
        script_path("validate_style_teacher.py"),
        "--teacher-adapter",
        str(repository_root() / teacher["adapter"]),
        "--teacher-id",
        teacher["teacher_id"],
        "--teacher-guidance-scale",
        str(teacher["guidance_scale"]),
        "--output",
        str(manifest),
    ]


def cache_command(
    manifest: Path,
    prompt_cache: Path,
    cache_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        script_path("cache_teacher_trajectories.py"),
        "--teacher-manifest",
        str(manifest),
        "--prompt-cache",
        str(prompt_cache),
        "--output-dir",
        str(cache_dir),
        "--replicas-per-prompt",
        "2",
        "--shard-size",
        "64",
    ]


def training_command(
    *,
    cache_dir: Path,
    prompt_cache: Path,
    init_adapter: Path,
    target_steps: int,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        script_path("train_phased_distill_lora.py"),
        "--trajectory-cache",
        str(cache_dir),
        "--prompt-cache",
        str(prompt_cache),
        "--init-adapter",
        str(init_adapter),
        "--target-steps",
        str(target_steps),
        "--output-dir",
        str(output_dir),
    ]


def evaluation_generation_command(
    *,
    teacher_manifest: Path,
    student_adapter: Path,
    student_steps: int,
    evaluation_prompts: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        script_path("generate_evaluation_set.py"),
        "--teacher-manifest",
        str(teacher_manifest),
        "--student-adapter",
        str(student_adapter),
        "--student-steps",
        str(student_steps),
        "--evaluation-prompts",
        str(evaluation_prompts),
        "--output-dir",
        str(output_dir),
        "--prompt-limit",
        "30",
        "--seeds-per-prompt",
        "4",
    ]


def metric_command(
    *,
    generation_dir: Path,
    evaluation_prompts: Path,
    metrics_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        script_path("evaluate_distilled.py"),
        "--teacher-images",
        str(generation_dir / "teacher"),
        "--student-images",
        str(generation_dir / "student"),
        "--evaluation-prompts",
        str(evaluation_prompts),
        "--output-dir",
        str(metrics_dir),
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "full_distillation_summary.json"
    command_log: list[list[str]] = []
    state: dict[str, Any] = {
        "format_version": 1,
        "status": "PLANNED" if args.dry_run else "RUNNING",
        "stage": "initialize",
        "output_root": str(output_root),
        "prompt_cache": str(args.prompt_cache.resolve()),
        "evaluation_prompts": str(args.evaluation_prompts.resolve()),
        "teachers": [],
        "commands": command_log,
    }
    if args.dry_run:
        prompt_assets = {
            "status": "NOT_VALIDATED_IN_DRY_RUN",
            "expected_prompt_count": 627,
            "expected_evaluation_prompt_count": 30,
        }
    else:
        prompt_assets = validate_prompt_assets(
            args.prompt_cache.resolve(), args.evaluation_prompts.resolve()
        )
    state["prompt_assets"] = prompt_assets
    update_state(state_path, state, "validate_teachers")

    contexts: list[dict[str, Any]] = []
    for teacher in TEACHERS:
        teacher_root = output_root / teacher["teacher_id"]
        teacher_root.mkdir(parents=True, exist_ok=True)
        manifest = teacher_root / "teacher_manifest.json"
        if args.rerun_completed or not matching_pass(
            manifest, teacher_id=teacher["teacher_id"]
        ):
            run_command(
                validation_command(teacher, manifest),
                dry_run=args.dry_run,
                command_log=command_log,
            )
        adapter = repository_root() / teacher["adapter"]
        if not args.dry_run:
            adapter = Path(read_json(manifest)["adapter_dir"])
        context = {
            **teacher,
            "teacher_root": teacher_root,
            "teacher_manifest": manifest,
            "teacher_adapter": adapter,
            "trajectory_cache": teacher_root / "trajectory_cache_v1",
            "student_4step": teacher_root / "student_4step",
            "student_2step": teacher_root / "student_2step",
        }
        contexts.append(context)
        state["teachers"].append(
            {
                "teacher_id": teacher["teacher_id"],
                "teacher_manifest": str(manifest),
                "guidance_scale": teacher["guidance_scale"],
            }
        )
    update_state(state_path, state, "cache_teacher_trajectories")

    # Cache both teachers before training either student so all expensive
    # teacher-only work is complete and resumable.
    for context in contexts:
        cache_manifest = context["trajectory_cache"] / "cache_manifest.json"
        if args.rerun_completed or not matching_pass(
            cache_manifest, trajectory_count=1_254
        ):
            run_command(
                cache_command(
                    context["teacher_manifest"],
                    args.prompt_cache.resolve(),
                    context["trajectory_cache"],
                ),
                dry_run=args.dry_run,
                command_log=command_log,
            )
    update_state(state_path, state, "train_4step_students")

    for context in contexts:
        metadata = context["student_4step"] / "run_metadata.json"
        if args.rerun_completed or not matching_pass(
            metadata,
            target_inference_steps=4,
            optimizer_steps=2_000,
        ):
            run_command(
                training_command(
                    cache_dir=context["trajectory_cache"],
                    prompt_cache=args.prompt_cache.resolve(),
                    init_adapter=context["teacher_adapter"],
                    target_steps=4,
                    output_dir=context["student_4step"],
                ),
                dry_run=args.dry_run,
                command_log=command_log,
            )
    update_state(state_path, state, "evaluate_4step_students")

    failed_four_step: list[dict[str, Any]] = []
    for context in contexts:
        evaluation_root = context["teacher_root"] / "evaluation_4step"
        generation_dir = evaluation_root / "images"
        generation_summary = generation_dir / "generation_summary.json"
        if args.rerun_completed or not matching_pass(
            generation_summary,
            student_steps=4,
            image_count_per_model=120,
        ):
            run_command(
                evaluation_generation_command(
                    teacher_manifest=context["teacher_manifest"],
                    student_adapter=context["student_4step"] / "best_adapter",
                    student_steps=4,
                    evaluation_prompts=args.evaluation_prompts.resolve(),
                    output_dir=generation_dir,
                ),
                dry_run=args.dry_run,
                command_log=command_log,
            )
        metrics_dir = evaluation_root / "metrics"
        metrics_summary = metrics_dir / "evaluation_summary.json"
        if args.rerun_completed or not matching_pass(metrics_summary):
            run_command(
                metric_command(
                    generation_dir=generation_dir,
                    evaluation_prompts=args.evaluation_prompts.resolve(),
                    metrics_dir=metrics_dir,
                ),
                dry_run=args.dry_run,
                command_log=command_log,
            )
        if not args.dry_run:
            metrics = read_json(metrics_summary)
            if metrics.get("status") != "PASS":
                failed_four_step.append(
                    {
                        "teacher_id": context["teacher_id"],
                        "metrics": str(metrics_summary),
                        "gates": metrics.get("gates"),
                    }
                )
    if failed_four_step:
        state["failed_4step_quality_gates"] = failed_four_step
        update_state(state_path, state, "4step_quality_gate", "STOPPED")
        raise QualityGateError(
            "At least one 4-step student failed the quality gate. "
            f"See {state_path}; 2-step training was not started."
        )

    update_state(state_path, state, "train_2step_students")
    for context in contexts:
        metadata = context["student_2step"] / "run_metadata.json"
        if args.rerun_completed or not matching_pass(
            metadata,
            target_inference_steps=2,
            optimizer_steps=10_000,
        ):
            run_command(
                training_command(
                    cache_dir=context["trajectory_cache"],
                    prompt_cache=args.prompt_cache.resolve(),
                    init_adapter=context["student_4step"] / "best_adapter",
                    target_steps=2,
                    output_dir=context["student_2step"],
                ),
                dry_run=args.dry_run,
                command_log=command_log,
            )
    if args.skip_final_2step_evaluation:
        update_state(state_path, state, "complete", "PASS")
        return state

    update_state(state_path, state, "evaluate_2step_students")
    failed_two_step: list[dict[str, Any]] = []
    for context in contexts:
        evaluation_root = context["teacher_root"] / "evaluation_2step"
        generation_dir = evaluation_root / "images"
        generation_summary = generation_dir / "generation_summary.json"
        if args.rerun_completed or not matching_pass(
            generation_summary,
            student_steps=2,
            image_count_per_model=120,
        ):
            run_command(
                evaluation_generation_command(
                    teacher_manifest=context["teacher_manifest"],
                    student_adapter=context["student_2step"] / "best_adapter",
                    student_steps=2,
                    evaluation_prompts=args.evaluation_prompts.resolve(),
                    output_dir=generation_dir,
                ),
                dry_run=args.dry_run,
                command_log=command_log,
            )
        metrics_dir = evaluation_root / "metrics"
        metrics_summary = metrics_dir / "evaluation_summary.json"
        if args.rerun_completed or not matching_pass(metrics_summary):
            run_command(
                metric_command(
                    generation_dir=generation_dir,
                    evaluation_prompts=args.evaluation_prompts.resolve(),
                    metrics_dir=metrics_dir,
                ),
                dry_run=args.dry_run,
                command_log=command_log,
            )
        if not args.dry_run:
            metrics = read_json(metrics_summary)
            if metrics.get("status") != "PASS":
                failed_two_step.append(
                    {
                        "teacher_id": context["teacher_id"],
                        "metrics": str(metrics_summary),
                        "gates": metrics.get("gates"),
                    }
                )
    if failed_two_step:
        state["failed_2step_quality_gates"] = failed_two_step
        update_state(state_path, state, "complete", "QUALITY_GATE_FAILED")
    else:
        update_state(state_path, state, "complete", "PASS")
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return state


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except QualityGateError as error:
        print(f"QUALITY GATE STOP: {error}", file=sys.stderr)
        return 2
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
