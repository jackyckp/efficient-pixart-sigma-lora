"""Preserve the pre-interruption best adapter across checkpoint resume."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from scripts.distillation import train_phased_distill_lora_impl as _impl
from scripts.distillation.common import write_json
from scripts.distillation.train_phased_distill_lora_v2 import *  # noqa: F401,F403


_save_checkpoint_without_history_guard = _impl._save_checkpoint
_train_without_history_guard = _impl.train


def existing_best_interval_loss(output_dir: Path) -> float:
    """Read the loss of the adapter currently promoted as stage best."""
    selection_path = output_dir / "best_checkpoint.json"
    if not selection_path.is_file():
        return math.inf
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        checkpoint = Path(selection["checkpoint"])
        metadata = json.loads(
            (checkpoint / "checkpoint_metadata.json").read_text(encoding="utf-8")
        )
        value = float(metadata["interval_mean_loss"])
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return math.inf
    return value if math.isfinite(value) else math.inf


def _save_checkpoint(*, best: bool, output_dir: Path, metadata: dict, **kwargs):
    """Reject a resumed-run promotion when an older adapter is still better."""
    prior_best = existing_best_interval_loss(output_dir)
    current = float(metadata["interval_mean_loss"])
    promote = best and current < prior_best
    return _save_checkpoint_without_history_guard(
        best=promote,
        output_dir=output_dir,
        metadata=metadata,
        **kwargs,
    )


def train(args: Any) -> dict[str, Any]:
    """Train normally, then report the true best loss across all resume runs."""
    result = _train_without_history_guard(args)
    output_dir = args.output_dir.resolve()
    historical_best = existing_best_interval_loss(output_dir)
    if historical_best < float(result["best_interval_mean_loss"]):
        result["best_interval_mean_loss"] = historical_best
        write_json(output_dir / "run_metadata.json", result)
    return result


_impl._save_checkpoint = _save_checkpoint
_impl.train = train

