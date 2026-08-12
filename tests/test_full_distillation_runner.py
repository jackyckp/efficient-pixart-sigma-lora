from __future__ import annotations

import json
from pathlib import Path

from scripts.distillation.run_full_two_teacher_distillation import (
    TEACHERS,
    build_parser,
    matching_pass,
    run,
    training_command,
)


def test_matching_pass_requires_status_and_expected_values(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "target_inference_steps": 4,
                "optimizer_steps": 2_000,
            }
        ),
        encoding="utf-8",
    )
    assert matching_pass(
        path, target_inference_steps=4, optimizer_steps=2_000
    )
    assert not matching_pass(path, target_inference_steps=2)


def test_training_command_keeps_stage_specific_adapter_and_steps(
    tmp_path: Path,
) -> None:
    command = training_command(
        cache_dir=tmp_path / "cache",
        prompt_cache=tmp_path / "prompts.pt",
        init_adapter=tmp_path / "student_4step" / "best_adapter",
        target_steps=2,
        output_dir=tmp_path / "student_2step",
    )
    assert command[command.index("--target-steps") + 1] == "2"
    assert command[command.index("--init-adapter") + 1].endswith(
        "student_4step\\best_adapter"
    )


def test_full_runner_dry_run_plans_both_complete_chains(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--dry-run", "--output-root", str(tmp_path / "outputs")]
    )
    state = run(args)
    assert state["status"] == "PASS"
    assert state["stage"] == "complete"
    assert len(state["teachers"]) == len(TEACHERS) == 2
    assert len(state["commands"]) == 16
    commands = [" ".join(command) for command in state["commands"]]
    assert sum("--target-steps 4" in command for command in commands) == 2
    assert sum("--target-steps 2" in command for command in commands) == 2
    assert sum("--student-steps 4" in command for command in commands) == 2
    assert sum("--student-steps 2" in command for command in commands) == 2
    summary = json.loads(
        (tmp_path / "outputs" / "full_distillation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "PASS"
