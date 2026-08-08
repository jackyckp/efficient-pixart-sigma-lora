from __future__ import annotations

import json
from pathlib import Path

from scripts.distillation.run_teacher_a_extended_6k_then_2step import (
    build_parser,
    run,
    seed_previous_best,
)


def test_seed_previous_best_keeps_formally_evaluated_4k_candidate(
    tmp_path: Path,
) -> None:
    source_adapter = tmp_path / "source" / "best_adapter"
    source_adapter.mkdir(parents=True)
    (source_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (source_adapter / "adapter_model.safetensors").write_bytes(b"weights")
    source_checkpoint = tmp_path / "source" / "checkpoints" / "step_004000"
    source_checkpoint.mkdir(parents=True)
    (source_checkpoint / "checkpoint_metadata.json").write_text(
        json.dumps({"interval_mean_loss": 0.044}), encoding="utf-8"
    )
    output = tmp_path / "new"
    seed_previous_best(
        output_dir=output,
        source_adapter=source_adapter,
        source_checkpoint=source_checkpoint,
    )
    assert (output / "best_adapter" / "adapter_model.safetensors").read_bytes() == b"weights"
    selection = json.loads(
        (output / "best_checkpoint.json").read_text(encoding="utf-8")
    )
    assert selection["optimizer_step"] == 4_000
    assert selection["checkpoint"] == str(source_checkpoint)


def test_dry_run_plans_isolated_4k_to_6k_then_7k_chain(
    tmp_path: Path,
    capsys,
) -> None:
    args = build_parser().parse_args(
        ["--dry-run", "--output-root", str(tmp_path / "experiment_6k")]
    )
    state = run(args)
    output = capsys.readouterr().out
    assert state["status"] == "PLANNED"
    assert "step_004000" in output
    assert "--target-steps 4" in output
    assert "--max-train-steps 6000" in output
    assert "--checkpoint-every-steps 500" in output
    assert "--target-steps 2" in output
    assert "--max-train-steps 7000" in output
    assert "--checkpoint-every-steps 1000" in output
    assert "teacher_a_extend6k_then2step" not in state["source_4step_checkpoint"]
    assert str(tmp_path / "experiment_6k") in state["extended_6k_output"]

