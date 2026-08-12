from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.distillation import train_phased_distill_lora_v3 as trainer


def write_existing_best(output_dir: Path, loss: float) -> None:
    checkpoint = output_dir / "checkpoints" / "step_001000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint_metadata.json").write_text(
        json.dumps({"interval_mean_loss": loss}), encoding="utf-8"
    )
    (output_dir / "best_checkpoint.json").write_text(
        json.dumps({"checkpoint": str(checkpoint)}), encoding="utf-8"
    )


def test_existing_best_interval_loss_survives_resume(tmp_path: Path) -> None:
    write_existing_best(tmp_path, 0.05)
    assert trainer.existing_best_interval_loss(tmp_path) == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("resumed_interval_loss", "expected_promotion"),
    ((0.06, False), (0.04, True)),
)
def test_checkpoint_promotion_compares_against_pre_resume_best(
    tmp_path: Path,
    monkeypatch,
    resumed_interval_loss: float,
    expected_promotion: bool,
) -> None:
    write_existing_best(tmp_path, 0.05)
    captured = {}

    def fake_save_checkpoint(**kwargs):
        captured.update(kwargs)
        return tmp_path / "saved"

    monkeypatch.setattr(
        trainer,
        "_save_checkpoint_without_history_guard",
        fake_save_checkpoint,
    )
    trainer._save_checkpoint(
        best=True,
        output_dir=tmp_path,
        metadata={"interval_mean_loss": resumed_interval_loss},
    )
    assert captured["best"] is expected_promotion

