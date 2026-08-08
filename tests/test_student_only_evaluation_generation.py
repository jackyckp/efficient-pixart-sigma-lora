from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.distillation.generate_student_evaluation_set import (
    validate_teacher_references,
)


def test_teacher_reference_reuse_requires_exact_prompt_seed_metadata(
    tmp_path: Path,
) -> None:
    record = {
        "prompt_id": "eval-01",
        "prompt": "pine on a cliff",
        "seed": 10011,
        "filename": "eval-01_seed10011.png",
    }
    image = tmp_path / record["filename"]
    image.write_bytes(b"png")
    image.with_suffix(".json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "prompt_id": record["prompt_id"],
                "prompt": record["prompt"],
                "seed": record["seed"],
                "num_inference_steps": 20,
                "transformer_forward_calls": 20,
            }
        ),
        encoding="utf-8",
    )
    validate_teacher_references(tmp_path, [record])
    bad = dict(record, seed=10012)
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_teacher_references(tmp_path, [bad])

