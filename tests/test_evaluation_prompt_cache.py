from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.distillation.common import COMPONENT_MODEL, MAX_SEQUENCE_LENGTH
from scripts.distillation.evaluation_prompt_cache import (
    CACHE_ROLE,
    EMBEDDING_DIM,
    load_evaluation_prompt_cache,
    prompt_set_fingerprint,
)


def _write_cache(path: Path, records: list[dict[str, str]]) -> None:
    torch.save(
        {
            "format_version": 1,
            "cache_role": CACHE_ROLE,
            "prompt_set_fingerprint": prompt_set_fingerprint(records),
            "prompt_ids": [record["prompt_id"] for record in records],
            "prompts": [record["prompt"] for record in records],
            "prompt_embeds": torch.zeros(
                (len(records), MAX_SEQUENCE_LENGTH, EMBEDDING_DIM), dtype=torch.float16
            ),
            "attention_masks": torch.ones(
                (len(records), MAX_SEQUENCE_LENGTH), dtype=torch.int64
            ),
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "text_encoder_model": COMPONENT_MODEL,
        },
        path,
    )


def test_cache_loads_requested_rows_without_t5(tmp_path: Path) -> None:
    records = [
        {"prompt_id": "eval/a", "prompt": "pine tree"},
        {"prompt_id": "eval/b", "prompt": "misty mountain"},
    ]
    cache_path = tmp_path / "evaluation.pt"
    _write_cache(cache_path, records)

    features = load_evaluation_prompt_cache(cache_path, [records[1], records[0]])

    assert list(features) == ["eval/b", "eval/a"]
    assert features["eval/a"][0].shape == (1, 300, 4096)
    assert features["eval/a"][1].shape == (1, 300)


def test_cache_rejects_changed_prompt_text(tmp_path: Path) -> None:
    records = [{"prompt_id": "eval/a", "prompt": "pine tree"}]
    cache_path = tmp_path / "evaluation.pt"
    _write_cache(cache_path, records)

    with pytest.raises(ValueError, match="text mismatch"):
        load_evaluation_prompt_cache(
            cache_path, [{"prompt_id": "eval/a", "prompt": "changed"}]
        )


def test_generation_parsers_default_to_persistent_cache() -> None:
    from scripts.distillation.generate_evaluation_set_v2 import build_parser as full_parser
    from scripts.distillation.generate_student_evaluation_set_v2 import (
        build_parser as student_parser,
    )

    assert any(action.dest == "evaluation_prompt_cache" for action in full_parser()._actions)
    assert any(action.dest == "evaluation_prompt_cache" for action in student_parser()._actions)
