"""Correct canonical five-dimensional trajectory shard loading."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from scripts.distillation import train_phased_distill_lora_impl as _impl
from scripts.distillation.common import LATENT_SHAPE, sha256_file
from scripts.distillation.train_phased_distill_lora_impl import *  # noqa: F401,F403


def load_trajectory_cache(
    cache_dir: Path,
    *,
    allow_partial: bool,
    expected_prompt_fingerprint: str,
):
    manifest = json.loads(
        (cache_dir / "cache_manifest.json").read_text(encoding="utf-8")
    )
    allowed = {"PASS", "PARTIAL"} if allow_partial else {"PASS"}
    if manifest.get("status") not in allowed:
        raise ValueError(
            f"Trajectory cache status must be {sorted(allowed)}, "
            f"got {manifest.get('status')!r}."
        )
    if manifest.get("prompt_bank_fingerprint") != expected_prompt_fingerprint:
        raise ValueError("Trajectory/prompt cache fingerprint mismatch.")
    if manifest.get("states_per_trajectory") != 21:
        raise ValueError("Trajectory cache must contain 21 states per record.")
    states_parts = []
    prompt_parts = []
    seed_parts = []
    for shard in manifest.get("shards", []):
        path = (cache_dir / shard["file"]).resolve()
        try:
            path.relative_to(cache_dir.resolve())
        except ValueError as error:
            raise ValueError(f"Unsafe trajectory shard path: {path}") from error
        if not path.is_file() or sha256_file(path) != shard["sha256"]:
            raise ValueError(f"Trajectory shard SHA mismatch: {path}")
        tensors = load_file(path, device="cpu")
        states = tensors["states"]
        if states.ndim != 5 or states.shape[1:] != (21, *LATENT_SHAPE):
            raise ValueError(
                f"Trajectory states must be [N,21,4,64,64], got "
                f"{tuple(states.shape)} in {path}."
            )
        if states.dtype != torch.float16 or not bool(torch.isfinite(states).all()):
            raise ValueError(f"Invalid trajectory states values in {path}.")
        states_parts.append(states)
        prompt_parts.append(tensors["prompt_indices"].to(torch.int64))
        seed_parts.append(tensors["seeds"].to(torch.int64))
    if not states_parts:
        raise ValueError("Trajectory cache has no shards.")
    states = torch.cat(states_parts).contiguous()
    prompt_indices = torch.cat(prompt_parts).contiguous()
    seeds = torch.cat(seed_parts).contiguous()
    if len(states) != manifest.get("trajectory_count"):
        raise ValueError("Trajectory manifest count does not match shards.")
    return manifest, states, prompt_indices, seeds


_impl.load_trajectory_cache = load_trajectory_cache
train = _impl.train

