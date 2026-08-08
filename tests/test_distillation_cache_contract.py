from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.distillation.common import LATENT_SHAPE, sha256_file
from scripts.distillation.train_phased_distill_lora import load_trajectory_cache


def _write_manifest(
    cache_dir: Path,
    shard_name: str,
    fingerprint: str,
) -> None:
    shard = cache_dir / shard_name
    payload = {
        "format_version": 1,
        "status": "PASS",
        "prompt_bank_fingerprint": fingerprint,
        "states_per_trajectory": 21,
        "trajectory_count": 2,
        "shards": [
            {
                "file": shard_name,
                "sha256": sha256_file(shard),
                "start_index": 0,
                "count": 2,
            }
        ],
    }
    (cache_dir / "cache_manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_load_canonical_five_dimensional_trajectory_shard(tmp_path: Path) -> None:
    shard = tmp_path / "trajectories-00000.safetensors"
    save_file(
        {
            "states": torch.zeros((2, 21, *LATENT_SHAPE), dtype=torch.float16),
            "prompt_indices": torch.tensor([0, 1], dtype=torch.int64),
            "seeds": torch.tensor([11, 12], dtype=torch.int64),
            "replicas": torch.tensor([0, 0], dtype=torch.int64),
        },
        shard,
    )
    _write_manifest(tmp_path, shard.name, "prompt-fingerprint")
    manifest, states, prompt_indices, seeds = load_trajectory_cache(
        tmp_path,
        allow_partial=False,
        expected_prompt_fingerprint="prompt-fingerprint",
    )
    assert manifest["status"] == "PASS"
    assert states.shape == (2, 21, 4, 64, 64)
    assert prompt_indices.tolist() == [0, 1]
    assert seeds.tolist() == [11, 12]


def test_trajectory_cache_rejects_path_traversal(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-trajectory.safetensors"
    save_file(
        {
            "states": torch.zeros((2, 21, *LATENT_SHAPE), dtype=torch.float16),
            "prompt_indices": torch.tensor([0, 1]),
            "seeds": torch.tensor([1, 2]),
        },
        outside,
    )
    _write_manifest(tmp_path, "../outside-trajectory.safetensors", "fingerprint")
    with pytest.raises(ValueError, match="Unsafe trajectory shard path"):
        load_trajectory_cache(
            tmp_path,
            allow_partial=False,
            expected_prompt_fingerprint="fingerprint",
        )
