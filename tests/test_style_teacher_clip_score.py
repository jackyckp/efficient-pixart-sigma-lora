from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.evaluation.evaluate_style_teacher_clip_score import (
    discover_images,
    load_source_metadata,
    rows_from_scores,
)


def make_source(root: Path) -> None:
    root.mkdir()
    (root / "individual").mkdir()
    (root / "official_base_model.png").write_bytes(b"base")
    (root / "evaluation_metadata.json").write_text(
        json.dumps({"status": "PASS", "prompt": "new palm prompt"}),
        encoding="utf-8",
    )
    for rank in (4, 8, 16, 32):
        for step in (1000, 2000):
            Image.new("RGB", (2, 2)).save(
                root / "individual" / f"step_{step:06d}_rank_{rank}.png"
            )


def test_discovers_base_and_common_rank_images(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_source(source)
    items = discover_images(source, (4, 8, 16, 32), 2)
    assert len(items) == 9
    assert items[0].step == 0
    assert items[0].rank is None
    assert [(item.step, item.rank) for item in items[1:5]] == [
        (1000, 4),
        (1000, 8),
        (1000, 16),
        (1000, 32),
    ]


def test_rejects_inconsistent_rank_steps(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_source(source)
    (source / "individual" / "step_002000_rank_32.png").unlink()
    with pytest.raises(ValueError, match="different image steps"):
        discover_images(source, (4, 8, 16, 32), 2)


def test_metadata_and_rows_are_strict(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_source(source)
    assert load_source_metadata(source)["prompt"] == "new palm prompt"
    items = discover_images(source, (4, 8, 16, 32), 2)
    rows = rows_from_scores(items, [0.3] + [0.4] * 8, (4, 8, 16, 32))
    assert rows[0] == {"step": 0, "official_base": 0.3}
    assert rows[1]["rank_32"] == 0.4

