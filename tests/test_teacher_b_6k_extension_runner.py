from __future__ import annotations

from pathlib import Path

from scripts.distillation.run_teacher_b_extended_6k_then_2step import (
    build_parser,
    run,
)


def test_dry_run_plans_isolated_teacher_b_4k_to_6k_then_7k_chain(
    tmp_path: Path,
    capsys,
) -> None:
    output_root = tmp_path / "teacher_b_6k"
    state = run(
        build_parser().parse_args(
            ["--dry-run", "--output-root", str(output_root)]
        )
    )
    output = capsys.readouterr().out

    assert state["status"] == "PLANNED"
    assert "teacher_b_extend4k_then2step" in state["source_4step_checkpoint"]
    assert "step_004000" in output
    assert "--target-steps 4" in output
    assert "--max-train-steps 6000" in output
    assert "--checkpoint-every-steps 500" in output
    assert "--target-steps 2" in output
    assert "--max-train-steps 7000" in output
    assert "--checkpoint-every-steps 2000" in output
    assert str(output_root) in state["extended_6k_output"]
    assert "teacher_b_extend4k_then2step" not in state["extended_6k_output"]
