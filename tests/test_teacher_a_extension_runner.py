from __future__ import annotations

import json
from pathlib import Path

from scripts.distillation.run_teacher_a_extended_4step_then_2step import (
    build_parser,
    latest_checkpoint,
    run,
    training_command,
)


def make_checkpoint(
    output: Path,
    step: int,
    *,
    target_steps: int,
    max_train_steps: int,
) -> Path:
    checkpoint = output / "checkpoints" / f"step_{step:06d}"
    adapter = checkpoint / "lora_adapter"
    adapter.mkdir(parents=True)
    (checkpoint / "training_state.pt").write_bytes(b"state")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "status": "CHECKPOINT",
                "target_inference_steps": target_steps,
                "max_train_steps": max_train_steps,
                "optimizer_step": step,
                "rank": 16,
                "lora_alpha": 16,
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


def test_extension_checkpoint_selection_uses_custom_terminal_step(
    tmp_path: Path,
) -> None:
    output = tmp_path / "four"
    make_checkpoint(output, 2_500, target_steps=4, max_train_steps=4_000)
    expected = make_checkpoint(
        output, 3_500, target_steps=4, max_train_steps=4_000
    )
    make_checkpoint(output, 4_000, target_steps=4, max_train_steps=4_000)
    assert latest_checkpoint(
        output, target_steps=4, max_train_steps=4_000
    ) == expected.resolve()


def test_stage_commands_lock_requested_steps_and_checkpoint_intervals(
    tmp_path: Path,
) -> None:
    command = training_command(
        trajectory_cache=tmp_path / "cache",
        prompt_cache=tmp_path / "prompts.pt",
        init_adapter=tmp_path / "four" / "best_adapter",
        output_dir=tmp_path / "two",
        target_steps=2,
        max_train_steps=7_000,
        checkpoint_every=1_000,
        learning_rate=2e-6,
    )
    assert command[command.index("--target-steps") + 1] == "2"
    assert command[command.index("--max-train-steps") + 1] == "7000"
    assert command[command.index("--checkpoint-every-steps") + 1] == "1000"
    assert "--resume-from" not in command


def test_dry_run_plans_isolated_four_then_two_chain(
    tmp_path: Path,
    capsys,
) -> None:
    args = build_parser().parse_args(
        ["--dry-run", "--output-root", str(tmp_path / "experiment")]
    )
    state = run(args)
    output = capsys.readouterr().out
    assert state["status"] == "PLANNED"
    assert "--target-steps 4" in output
    assert "--max-train-steps 4000" in output
    assert "--checkpoint-every-steps 500" in output
    assert "--target-steps 2" in output
    assert "--max-train-steps 7000" in output
    assert "--checkpoint-every-steps 1000" in output
    assert "outputs\\distillation\\plant_n209_r16_step10200" in output
    assert str(tmp_path / "experiment") in state["extended_4step_output"]

