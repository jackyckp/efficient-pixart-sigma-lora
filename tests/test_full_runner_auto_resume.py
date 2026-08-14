from __future__ import annotations

import json
from pathlib import Path

import scripts.distillation.run_full_two_teacher_distillation as runner
from scripts.distillation import run_full_two_teacher_distillation_impl as resume_impl


FINGERPRINT = "f" * 64


def make_checkpoint(
    output_dir: Path,
    step: int,
    *,
    target_steps: int,
    max_train_steps: int,
    valid: bool = True,
) -> Path:
    checkpoint = output_dir / "checkpoints" / f"step_{step:06d}"
    adapter = checkpoint / "lora_adapter"
    adapter.mkdir(parents=True)
    (checkpoint / "training_state.pt").write_bytes(b"optimizer-state")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    metadata = {
        "status": "CHECKPOINT",
        "run_role": "phased_joint_lora_student",
        "target_inference_steps": target_steps,
        "max_train_steps": max_train_steps,
        "optimizer_step": step,
        "rank": 16,
        "lora_alpha": 16,
        "prompt_bank_fingerprint": FINGERPRINT,
    }
    if not valid:
        metadata["optimizer_step"] = step - 1
    (checkpoint / "checkpoint_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return checkpoint


def test_latest_resume_checkpoint_selects_newest_complete_stage_state(
    tmp_path: Path,
) -> None:
    output = tmp_path / "student_2step"
    make_checkpoint(output, 1_000, target_steps=2, max_train_steps=10_000)
    expected = make_checkpoint(
        output, 7_000, target_steps=2, max_train_steps=10_000
    )
    make_checkpoint(
        output,
        8_000,
        target_steps=2,
        max_train_steps=10_000,
        valid=False,
    )
    # A final checkpoint cannot be passed to the trainer because it already
    # reached max_train_steps; recovery deliberately repeats the last interval.
    make_checkpoint(output, 10_000, target_steps=2, max_train_steps=10_000)
    assert runner.latest_resume_checkpoint(output, target_steps=2) == expected


def test_four_step_final_checkpoint_falls_back_to_previous_interval(
    tmp_path: Path,
) -> None:
    output = tmp_path / "student_4step"
    expected = make_checkpoint(
        output, 1_500, target_steps=4, max_train_steps=2_000
    )
    make_checkpoint(output, 2_000, target_steps=4, max_train_steps=2_000)
    assert runner.latest_resume_checkpoint(output, target_steps=4) == expected


def test_training_command_automatically_appends_resume_checkpoint(
    tmp_path: Path,
) -> None:
    output = tmp_path / "student_2step"
    checkpoint = make_checkpoint(
        output, 6_000, target_steps=2, max_train_steps=10_000
    )
    command = runner.training_command(
        cache_dir=tmp_path / "cache",
        prompt_cache=tmp_path / "prompts.pt",
        init_adapter=tmp_path / "student_4step" / "best_adapter",
        target_steps=2,
        output_dir=output,
    )
    assert command[command.index("--resume-from") + 1] == str(checkpoint.resolve())


def test_explicit_rerun_mode_disables_automatic_resume(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "student_2step"
    make_checkpoint(output, 6_000, target_steps=2, max_train_steps=10_000)
    monkeypatch.setattr(resume_impl, "_AUTO_RESUME_ENABLED", False)
    command = runner.training_command(
        cache_dir=tmp_path / "cache",
        prompt_cache=tmp_path / "prompts.pt",
        init_adapter=tmp_path / "student_4step" / "best_adapter",
        target_steps=2,
        output_dir=output,
    )
    assert "--resume-from" not in command

