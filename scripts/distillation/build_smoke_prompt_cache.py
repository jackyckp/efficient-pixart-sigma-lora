#!/usr/bin/env python3
"""Build a four-prompt format-v2 cache without loading T5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from scripts.distillation.common import (  # noqa: E402
    MANIFEST_FINGERPRINT,
    fingerprint_records,
    repository_root,
    save_prompt_bank,
    write_json,
)
from scripts.training.train_local_latent_lora import load_latent_bundle  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description="Reuse four canonical plant embeddings for GPU smoke tests."
    )
    parser.add_argument(
        "--latent-bundle",
        type=Path,
        default=root / "data" / "archives" / "clean_latents_512.zip",
    )
    parser.add_argument(
        "--source-prompt-cache",
        type=Path,
        default=(
            root
            / "data"
            / "features"
            / "t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt"
        ),
    )
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--prompt-bank", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=4)
    return parser


def build_smoke_cache(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive.")
    bundle = load_latent_bundle(args.latent_bundle)
    manifest_by_id = {row["sample_id"]: row for row in bundle.manifest}
    plant_ids = sorted(
        sample_id
        for sample_id in bundle.sample_ids
        if sample_id.startswith("plant/")
    )[: args.num_prompts]
    if len(plant_ids) != args.num_prompts:
        raise ValueError("Latent bundle does not contain enough plant prompts.")
    source = torch.load(
        args.source_prompt_cache, map_location="cpu", weights_only=True
    )
    if source.get("manifest_fingerprint") != MANIFEST_FINGERPRINT:
        raise ValueError("Source prompt cache fingerprint mismatch.")
    source_index = {
        sample_id: index for index, sample_id in enumerate(source["sample_ids"])
    }
    missing = [sample_id for sample_id in plant_ids if sample_id not in source_index]
    if missing:
        raise ValueError(f"Source prompt cache misses smoke IDs: {missing}")
    records = tuple(
        {
            "prompt_id": f"{sample_id}::original",
            "source_sample_id": sample_id,
            "variant": "original",
            "prompt": manifest_by_id[sample_id]["caption"],
            "category": "plant",
            "training_only": True,
        }
        for sample_id in plant_ids
    )
    fingerprint = save_prompt_bank(args.prompt_bank, records)
    indices = torch.tensor(
        [source_index[sample_id] for sample_id in plant_ids], dtype=torch.long
    )
    cache = {
        "format_version": 2,
        "prompt_ids": [row["prompt_id"] for row in records],
        "source_sample_ids": plant_ids,
        "prompts": [row["prompt"] for row in records],
        "variants": ["original"] * len(records),
        "prompt_embeds": source["prompt_embeds"].index_select(0, indices),
        "attention_masks": source["attention_masks"].index_select(0, indices),
        "empty_prompt_embeds": source["empty_prompt_embeds"],
        "empty_prompt_attention_mask": source["empty_prompt_attention_mask"],
        "max_sequence_length": source["max_sequence_length"],
        "text_encoder_model": source["text_encoder_model"],
        "source_manifest_fingerprint": MANIFEST_FINGERPRINT,
        "prompt_bank_fingerprint": fingerprint,
        "prompt_bank_records_fingerprint": fingerprint_records(records),
        "smoke_only": True,
    }
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output_cache)
    result = {
        "status": "PASS",
        "smoke_only": True,
        "prompt_count": len(records),
        "prompt_ids": cache["prompt_ids"],
        "prompt_bank_fingerprint": fingerprint,
        "prompt_cache": str(args.output_cache.resolve()),
        "prompt_bank": str(args.prompt_bank.resolve()),
    }
    write_json(args.output_cache.with_suffix(".validation.json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_smoke_cache(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
