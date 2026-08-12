#!/usr/bin/env python3
"""Encode held-out evaluation prompts once and persist their T5 features."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.distillation.common import COMPONENT_MODEL, MAX_SEQUENCE_LENGTH  # noqa: E402
from scripts.distillation.evaluation_prompt_cache import (  # noqa: E402
    CACHE_ROLE,
    DEFAULT_CACHE,
    prompt_set_fingerprint,
    records_from_evaluation_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build reusable evaluation T5 cache.")
    parser.add_argument("--evaluation-prompts", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--component-model", default=COMPONENT_MODEL)
    parser.add_argument("--t5-gpu-memory", default="8GiB")
    parser.add_argument("--t5-cpu-memory", default="24GiB")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def build_cache(args: argparse.Namespace) -> dict[str, object]:
    records = records_from_evaluation_manifest(args.evaluation_prompts.resolve())
    output = args.output_cache.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Cache already exists (use --overwrite): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(
        args.component_model, subfolder="tokenizer", local_files_only=args.local_files_only
    )
    offload = output.parent / ".t5_offload_evaluation_cache"
    offload.mkdir(parents=True, exist_ok=True)
    encoder = T5EncoderModel.from_pretrained(
        args.component_model,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: args.t5_gpu_memory, "cpu": args.t5_cpu_memory},
        offload_folder=str(offload),
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).eval()
    embeddings, masks = [], []
    for record in records:
        tokens = tokenizer(
            [record["prompt"]], padding="max_length", max_length=MAX_SEQUENCE_LENGTH,
            truncation=True, return_attention_mask=True, return_tensors="pt",
        )
        input_device = encoder.get_input_embeddings().weight.device
        with torch.inference_mode():
            embedding = encoder(
                input_ids=tokens.input_ids.to(input_device),
                attention_mask=tokens.attention_mask.to(input_device),
            ).last_hidden_state.to("cpu", dtype=torch.float16)
        if not bool(torch.isfinite(embedding).all()):
            raise FloatingPointError(f"Non-finite embedding: {record['prompt_id']}")
        embeddings.append(embedding)
        masks.append(tokens.attention_mask.to("cpu", dtype=torch.int64))
    del encoder, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fingerprint = prompt_set_fingerprint(records)
    payload = {
        "format_version": 1,
        "cache_role": CACHE_ROLE,
        "prompt_set_fingerprint": fingerprint,
        "prompt_ids": [record["prompt_id"] for record in records],
        "prompts": [record["prompt"] for record in records],
        "prompt_embeds": torch.cat(embeddings, dim=0),
        "attention_masks": torch.cat(masks, dim=0),
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "text_encoder_model": args.component_model,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    result = {
        "status": "PASS", "output_cache": str(output), "prompt_count": len(records),
        "prompt_set_fingerprint": fingerprint,
        "prompt_embeds_shape": list(payload["prompt_embeds"].shape),
        "prompt_embeds_dtype": str(payload["prompt_embeds"].dtype),
        "attention_masks_shape": list(payload["attention_masks"].shape),
        "text_encoder_model": args.component_model,
    }
    output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    build_cache(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
