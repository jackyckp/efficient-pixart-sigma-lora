#!/usr/bin/env python3
"""Resume-safe two-teacher 20 -> 4 -> 2 distillation orchestrator."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from scripts.distillation.common import repository_root, write_json


TEACHERS = (
    {
        "id": "plant_n209_r16_step10200",
        "adapter": Path(
            "outputs/style_teacher/plant_n209_steps10200/r16_lr1e-05"
        ),
        "guidance": 1.0,
    },
    {
        "id": "teammate_plant209_step4000",
        "adapter": Path(
            "outputs/style_teacher/best_ink_wash_lora_plant209_step4000"
        ),
        "guidance": 1.5,
    },
)


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description="Run two independent PixArt phased LoRA distillations."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "outputs" / "distillation",
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
        "--stop-after",
        choices=("validate", "prompts", "cache", "4step", "2step", "evaluate"),
        default="evaluate",
    )
    parser.add_argument("--skip-quality-gate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    return parser


def _completed(path: Path, **expected: Any) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "PASS" and all(
        payload.get(key) == value for key, value in expected.items()
    )


def _run(command: list[str], *, dry_run: bool) -> None:
    print("COMMAND:")
    print(subprocess.list2cmdline(command))
    if not dry_run:
        subprocess.run(command, check=True)


def _script(name: str) -> str:
    return str(repository_root() / "scripts" / "distillation" / name)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = repository_root()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "status": "PLANNED" if args.dry_run else "RUNNING",
        "smoke": args.smoke,
        "stop_after": args.stop_after,
        "teachers": [],
    }
    for definition in TEACHERS:
        teacher = {**definition, "adapter": root / definition["adapter"]}
        teacher_root = output_root / teacher["id"]
        teacher_root.mkdir(parents=True, exist_ok=True)
        manifest = teacher_root / "teacher_manifest.json"
        if args.rerun_completed or not _completed(
            manifest, teacher_id=teacher["id"]
        ):
            command = [
                sys.executable,
                _script("validate_style_teacher.py"),
                "--teacher-adapter",
                str(teacher["adapter"]),
                "--teacher-id",
                teacher["id"],
                "--teacher-guidance-scale",
                str(teacher["guidance"]),
                "--output",
                str(manifest),
            ]
            if args.dry_run:
                command.append("--skip-fresh-load")
            _run(command, dry_run=args.dry_run)
        summary["teachers"].append(
            {
                "teacher_id": teacher["id"],
                "manifest": str(manifest),
                "guidance": teacher["guidance"],
            }
        )
    if args.stop_after == "validate":
        summary["status"] = "PASS"
        write_json(output_root / "orchestration_summary.json", summary)
        return summary

    prompt_bank = root / "data" / "distillation" / "plant_prompt_bank_v1.jsonl"
    prompt_ready = args.prompt_cache.is_file() and args.evaluation_prompts.is_file()
    if args.rerun_completed or not prompt_ready:
        command = [
            sys.executable,
            _script("build_distill_prompt_cache.py"),
            "--prompt-bank",
            str(prompt_bank),
            "--evaluation-prompts",
            str(args.evaluation_prompts),
            "--output-cache",
            str(args.prompt_cache),
        ]
        _run(command, dry_run=args.dry_run)
    if args.stop_after == "prompts":
        summary["status"] = "PASS"
        write_json(output_root / "orchestration_summary.json", summary)
        return summary

    for teacher_entry in summary["teachers"]:
        teacher_root = output_root / teacher_entry["teacher_id"]
        cache_dir = teacher_root / "trajectory_cache_v1"
        cache_manifest = cache_dir / "cache_manifest.json"
        cache_ok = _completed(cache_manifest)
        if args.smoke:
            cache_ok = cache_manifest.is_file()
        if args.rerun_completed or not cache_ok:
            command = [
                sys.executable,
                _script("cache_teacher_trajectories.py"),
                "--teacher-manifest",
                teacher_entry["manifest"],
                "--prompt-cache",
                str(args.prompt_cache),
                "--output-dir",
                str(cache_dir),
                "--replicas-per-prompt",
                "2",
            ]
            if args.smoke:
                command.extend(["--limit-trajectories", "8", "--shard-size", "8"])
            _run(command, dry_run=args.dry_run)
        teacher_entry["trajectory_cache"] = str(cache_dir)
    if args.stop_after == "cache":
        summary["status"] = "PASS"
        write_json(output_root / "orchestration_summary.json", summary)
        return summary

    for teacher_entry in summary["teachers"]:
        teacher_root = output_root / teacher_entry["teacher_id"]
        four_dir = teacher_root / "student_4step"
        if args.rerun_completed or not _completed(
            four_dir / "run_metadata.json", target_inference_steps=4
        ):
            command = [
                sys.executable,
                _script("train_phased_distill_lora.py"),
                "--trajectory-cache",
                teacher_entry["trajectory_cache"],
                "--prompt-cache",
                str(args.prompt_cache),
                "--init-adapter",
                str(Path(teacher_entry["manifest"]).parent),
                "--target-steps",
                "4",
                "--output-dir",
                str(four_dir),
            ]
            # The validator manifest directory is not itself an adapter. Replace
            # it with the source adapter recorded by validation.
            if not args.dry_run:
                manifest_data = json.loads(
                    Path(teacher_entry["manifest"]).read_text(encoding="utf-8")
                )
                init_index = command.index("--init-adapter") + 1
                command[init_index] = manifest_data["adapter_dir"]
            if args.smoke:
                command.extend(
                    [
                        "--max-train-steps",
                        "20",
                        "--checkpoint-every-steps",
                        "20",
                        "--allow-partial-cache",
                    ]
                )
            _run(command, dry_run=args.dry_run)
        teacher_entry["student_4step"] = str(four_dir)
    if args.stop_after == "4step" or args.smoke:
        summary["status"] = "PASS"
        write_json(output_root / "orchestration_summary.json", summary)
        return summary

    # The 4-step metric gate is deliberately external: generate/evaluate the
    # fixed suite, then continue only when evaluation_summary.json is PASS.
    if not args.skip_quality_gate:
        missing_gates = []
        for teacher_entry in summary["teachers"]:
            gate = (
                output_root
                / teacher_entry["teacher_id"]
                / "evaluation_4step"
                / "metrics"
                / "evaluation_summary.json"
            )
            if not _completed(gate):
                missing_gates.append(str(gate))
        if missing_gates:
            summary["status"] = "WAITING_FOR_4STEP_EVALUATION"
            summary["required_quality_gates"] = missing_gates
            write_json(output_root / "orchestration_summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return summary

    for teacher_entry in summary["teachers"]:
        teacher_root = output_root / teacher_entry["teacher_id"]
        two_dir = teacher_root / "student_2step"
        init_adapter = teacher_root / "student_4step" / "best_adapter"
        if args.rerun_completed or not _completed(
            two_dir / "run_metadata.json", target_inference_steps=2
        ):
            _run(
                [
                    sys.executable,
                    _script("train_phased_distill_lora.py"),
                    "--trajectory-cache",
                    teacher_entry["trajectory_cache"],
                    "--prompt-cache",
                    str(args.prompt_cache),
                    "--init-adapter",
                    str(init_adapter),
                    "--target-steps",
                    "2",
                    "--output-dir",
                    str(two_dir),
                ],
                dry_run=args.dry_run,
            )
        teacher_entry["student_2step"] = str(two_dir)
    summary["status"] = "PASS"
    write_json(output_root / "orchestration_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
