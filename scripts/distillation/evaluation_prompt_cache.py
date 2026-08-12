"""Persistent T5 features for the fixed distillation evaluation prompt set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from scripts.distillation.common import COMPONENT_MODEL, MAX_SEQUENCE_LENGTH, fingerprint_records

EMBEDDING_DIM = 4096
CACHE_ROLE = "distillation_evaluation_prompt_embeddings"
DEFAULT_CACHE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "features"
    / "distill_eval_t5_prompts30_len300_fp16_v1.pt"
)


def unique_prompt_records(records: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Return unique prompts while rejecting conflicting duplicate IDs."""
    prompts: dict[str, str] = {}
    for record in records:
        prompt_id, prompt = str(record["prompt_id"]), str(record["prompt"])
        previous = prompts.setdefault(prompt_id, prompt)
        if previous != prompt:
            raise ValueError(f"Prompt ID has conflicting text: {prompt_id}")
    return [
        {"prompt_id": prompt_id, "prompt": prompt}
        for prompt_id, prompt in prompts.items()
    ]


def prompt_set_fingerprint(records: Sequence[dict[str, Any]]) -> str:
    return fingerprint_records(unique_prompt_records(records))


def records_from_evaluation_manifest(path: str | Path) -> list[dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Evaluation prompt manifest contains no prompts.")
    return unique_prompt_records(prompts)


def load_evaluation_prompt_cache(
    cache_path: str | Path,
    requested_records: Sequence[dict[str, Any]],
    *,
    component_model: str = COMPONENT_MODEL,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Load, validate, and index a saved evaluation prompt cache."""
    path = Path(cache_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation prompt cache is missing: {path}\n"
            "Build it once with scripts/distillation/build_evaluation_prompt_cache.py."
        )
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(cache, dict):
        raise ValueError("Evaluation prompt cache must be a dictionary.")
    if cache.get("format_version") != 1 or cache.get("cache_role") != CACHE_ROLE:
        raise ValueError("Unsupported evaluation prompt cache format or role.")
    if cache.get("text_encoder_model") != component_model:
        raise ValueError("Evaluation prompt cache text encoder model mismatch.")
    if cache.get("max_sequence_length") != MAX_SEQUENCE_LENGTH:
        raise ValueError("Evaluation prompt cache sequence length mismatch.")

    prompt_ids, prompts = cache.get("prompt_ids"), cache.get("prompts")
    embeds, masks = cache.get("prompt_embeds"), cache.get("attention_masks")
    if not isinstance(prompt_ids, list) or not isinstance(prompts, list):
        raise ValueError("Evaluation prompt cache ID/text lists are invalid.")
    if len(prompt_ids) != len(set(prompt_ids)) or len(prompt_ids) != len(prompts):
        raise ValueError("Evaluation prompt cache IDs are duplicated or misaligned.")
    cache_records = [
        {"prompt_id": str(prompt_id), "prompt": str(prompt)}
        for prompt_id, prompt in zip(prompt_ids, prompts, strict=True)
    ]
    if cache.get("prompt_set_fingerprint") != prompt_set_fingerprint(cache_records):
        raise ValueError("Evaluation prompt cache fingerprint is corrupt.")
    if not isinstance(embeds, torch.Tensor) or embeds.shape != (
        len(prompt_ids), MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
    ):
        raise ValueError("Evaluation prompt embedding shape mismatch.")
    if embeds.dtype != torch.float16 or embeds.device.type != "cpu":
        raise ValueError("Evaluation prompt embeddings must be FP16 CPU tensors.")
    if not bool(torch.isfinite(embeds).all()):
        raise ValueError("Evaluation prompt embeddings contain non-finite values.")
    if not isinstance(masks, torch.Tensor) or masks.shape != (
        len(prompt_ids), MAX_SEQUENCE_LENGTH
    ):
        raise ValueError("Evaluation prompt attention mask shape mismatch.")
    if masks.dtype not in (torch.bool, torch.int64):
        raise ValueError("Evaluation prompt attention masks must be bool or int64.")
    if not bool(((masks == 0) | (masks == 1)).all()):
        raise ValueError("Evaluation prompt attention masks must be binary.")

    row_by_id = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
    output: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for record in unique_prompt_records(requested_records):
        prompt_id = record["prompt_id"]
        if prompt_id not in row_by_id:
            raise ValueError(f"Evaluation prompt cache is missing ID: {prompt_id}")
        row = row_by_id[prompt_id]
        if prompts[row] != record["prompt"]:
            raise ValueError(f"Evaluation prompt cache text mismatch: {prompt_id}")
        output[prompt_id] = (embeds[row : row + 1], masks[row : row + 1])
    return output
