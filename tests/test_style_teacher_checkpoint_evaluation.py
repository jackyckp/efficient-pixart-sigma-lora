from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.evaluation.generate_style_teacher_checkpoint_grids import (
    build_grid,
    discover_checkpoint_groups,
    parser,
)


def write_checkpoint(root: Path, rank: int, step: int, loss: float) -> None:
    checkpoint = root / f"r{rank}_lr1e-05" / "checkpoints" / f"step_{step:06d}"
    adapter = checkpoint / "lora_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "checkpoint_metadata.json").write_text(
        json.dumps({"loss": loss}), encoding="utf-8"
    )


def test_discovery_orders_steps_and_ranks(tmp_path: Path) -> None:
    for rank in (4, 8, 16, 32):
        write_checkpoint(tmp_path, rank, 1000, rank / 100)
        write_checkpoint(tmp_path, rank, 2000, rank / 100)
    groups = discover_checkpoint_groups(tmp_path, (4, 8, 16, 32), 2)
    assert list(groups) == [1000, 2000]
    assert [item.rank for item in groups[1000]] == [4, 8, 16, 32]


def test_discovery_rejects_a_missing_checkpoint(tmp_path: Path) -> None:
    for rank in (4, 8, 16, 32):
        write_checkpoint(tmp_path, rank, 1000, 0.1)
    for rank in (4, 8, 16):
        write_checkpoint(tmp_path, rank, 2000, 0.2)
    with pytest.raises(ValueError, match="identical checkpoint steps"):
        discover_checkpoint_groups(tmp_path, (4, 8, 16, 32), 2)


def test_grid_has_four_captioned_tiles(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    for rank in (4, 8, 16, 32):
        write_checkpoint(run_root, rank, 1000, 0.1)
    checkpoints = discover_checkpoint_groups(run_root, (4, 8, 16, 32), 1)[1000]
    output = tmp_path / "grid.png"
    build_grid(
        [Image.new("RGB", (8, 8), (rank, 0, 0)) for rank in (4, 8, 16, 32)],
        checkpoints,
        1000,
        parser().parse_args([]),
        output,
    )
    with Image.open(output) as grid:
        assert grid.size == (1024, 1166)


def test_parser_supports_official_base_model_comparison() -> None:
    args = parser().parse_args(
        ["--include-base-model", "--prompt-source-sample-id", "plant/220"]
    )
    assert args.include_base_model is True
    assert args.prompt_source_sample_id == "plant/220"

