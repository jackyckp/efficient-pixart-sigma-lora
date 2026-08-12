#!/usr/bin/env python3
"""Resume-safe entry point for the complete two-teacher workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scripts.distillation import run_full_two_teacher_distillation_impl as _impl
from scripts.distillation.run_full_two_teacher_distillation_impl import *  # noqa: F401,F403


_STEP_DIRECTORY = re.compile(r"^step_(\d{6})$")
_STAGE_MAX_STEPS = {4: 2_000, 2: 10_000}
_AUTO_RESUME_ENABLED = True
_training_command_without_resume = _impl.training_command
_run_without_resume_configuration = _impl.run


def _checkpoint_step_if_valid(
    checkpoint: Path,
    *,
    target_steps: int,
    max_train_steps: int,
) -> int | None:
    """Validate the inexpensive checkpoint contract used before subprocesses."""
    match = _STEP_DIRECTORY.fullmatch(checkpoint.name)
    if match is None or not checkpoint.is_dir():
        return None
    directory_step = int(match.group(1))
    # The trainer refuses to resume at global_step >= max_train_steps.  If the
    # process died after the final checkpoint but before run_metadata.json, use
    # the preceding checkpoint and safely repeat only the last interval.
    if not 0 < directory_step < max_train_steps:
        return None
    metadata_path = checkpoint / "checkpoint_metadata.json"
    state_path = checkpoint / "training_state.pt"
    adapter_dir = checkpoint / "lora_adapter"
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    required_files = (
        metadata_path,
        state_path,
        adapter_config,
        adapter_weights,
    )
    if any(not path.is_file() or path.stat().st_size == 0 for path in required_files):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "status": "CHECKPOINT",
        "run_role": "phased_joint_lora_student",
        "target_inference_steps": target_steps,
        "max_train_steps": max_train_steps,
        "optimizer_step": directory_step,
        "rank": 16,
        "lora_alpha": 16,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    fingerprint = metadata.get("prompt_bank_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        return None
    return directory_step


def latest_resume_checkpoint(
    output_dir: Path,
    *,
    target_steps: int,
    max_train_steps: int | None = None,
) -> Path | None:
    """Return the newest complete checkpoint that the trainer can continue."""
    if target_steps not in _STAGE_MAX_STEPS:
        raise ValueError(f"Unsupported target step count: {target_steps}")
    expected_max = _STAGE_MAX_STEPS[target_steps]
    if max_train_steps is None:
        max_train_steps = expected_max
    if max_train_steps != expected_max:
        raise ValueError(
            f"Expected max_train_steps={expected_max} for {target_steps}-step, "
            f"got {max_train_steps}."
        )
    checkpoint_root = output_dir.resolve() / "checkpoints"
    if not checkpoint_root.is_dir():
        return None
    valid: list[tuple[int, Path]] = []
    for candidate in checkpoint_root.iterdir():
        step = _checkpoint_step_if_valid(
            candidate,
            target_steps=target_steps,
            max_train_steps=max_train_steps,
        )
        if step is not None:
            valid.append((step, candidate.resolve()))
    return max(valid, default=(0, None), key=lambda item: item[0])[1]


def training_command(
    *,
    cache_dir: Path,
    prompt_cache: Path,
    init_adapter: Path,
    target_steps: int,
    output_dir: Path,
) -> list[str]:
    """Build a stage command and automatically attach its newest checkpoint."""
    command = _training_command_without_resume(
        cache_dir=cache_dir,
        prompt_cache=prompt_cache,
        init_adapter=init_adapter,
        target_steps=target_steps,
        output_dir=output_dir,
    )
    if _AUTO_RESUME_ENABLED:
        checkpoint = latest_resume_checkpoint(
            output_dir,
            target_steps=target_steps,
        )
        if checkpoint is not None:
            command.extend(("--resume-from", str(checkpoint)))
            print(
                f"AUTO-RESUME {target_steps}-step training from {checkpoint}",
                flush=True,
            )
    return command


def run(args: Any) -> dict[str, Any]:
    """Configure resume behavior and delegate to the tested orchestration."""
    global _AUTO_RESUME_ENABLED
    previous = _AUTO_RESUME_ENABLED
    _AUTO_RESUME_ENABLED = not bool(args.rerun_completed)
    try:
        return _run_without_resume_configuration(args)
    finally:
        _AUTO_RESUME_ENABLED = previous


# The implementation's ``run`` resolves ``training_command`` in its own module.
# Its imported ``main`` also resolves ``run`` there, so patch both boundaries.
_impl.training_command = training_command
_impl.run = run


if __name__ == "__main__":
    raise SystemExit(main())

