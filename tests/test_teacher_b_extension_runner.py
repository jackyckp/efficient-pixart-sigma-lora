from __future__ import annotations

import json
from pathlib import Path

from scripts.distillation.run_teacher_b_extended_4k_then_2step import (
    build_parser,
    run,
    seed_previous_best,
)


def test_seed_previous_best_preserves_accepted_teacher_b_2k(
    tmp_path: Path,
) -> None:
    source_adapter = tmp_path / "source" / "best_adapter"
    source_adapter.mkdir(parents=True)
    (source_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (source_adapter / "adapter_model.safetensors").write_bytes(b"teacher-b")
    source_checkpoint = tmp_path / "source" / "checkpoints" / "step_002000"
    source_checkpoint.mkdir(parents=True)
    (source_checkpoint / "checkpoint_metadata.json").write_text(
        json.dumps({"interval_mean_loss": 0.05}), encoding="utf-8"
    )
    output = tmp_path / "extended"
    seed_previous_best(
        output_dir=output,
        source_adapter=source_adapter,
        source_checkpoint=source_checkpoint,
    )
    assert (output / "best_adapter" / "adapter_model.safetensors").read_bytes() == b"teacher-b"
    selection = json.loads(
        (output / "best_checkpoint.json").read_text(encoding="utf-8")
    )
    assert selection["optimizer_step"] == 2_000
    assert selection["checkpoint"] == str(source_checkpoint)


def test_dry_run_plans_teacher_b_4k_and_2k_interval_student(
    tmp_path: Path,
    capsys,
) -> None:
    args = build_parser().parse_args(
        ["--dry-run", "--output-root", str(tmp_path / "teacher_b_experiment")]
    )
    state = run(args)
    output = capsys.readouterr().out
    assert state["status"] == "PLANNED"
    assert "teammate_plant209_step4000" in state["source_4step_checkpoint"]
    assert "step_002000" in output
    assert "--target-steps 4" in output
    assert "--max-train-steps 4000" in output
    assert "--checkpoint-every-steps 500" in output
    assert "--target-steps 2" in output
    assert "--max-train-steps 7000" in output
    assert "--checkpoint-every-steps 2000" in output
    assert str(tmp_path / "teacher_b_experiment") in state["extended_4step_output"]

