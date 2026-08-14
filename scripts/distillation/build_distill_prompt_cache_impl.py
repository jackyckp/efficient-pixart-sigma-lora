#!/usr/bin/env python3
"""Build the augmented plant prompt bank and its T5-XXL feature cache."""

from __future__ import annotations

import argparse
import gc
import json
import platform
from pathlib import Path
from typing import Any, Sequence

import torch

from scripts.distillation.common import (
    COMPONENT_MODEL,
    EMBEDDING_DIM,
    MANIFEST_FINGERPRINT,
    MAX_SEQUENCE_LENGTH,
    STYLE_TRIGGER,
    batched,
    build_prompt_records,
    fingerprint_records,
    repository_root,
    save_prompt_bank,
    write_json,
)
from scripts.training.train_local_latent_lora import load_latent_bundle


EVALUATION_SUBJECTS = (
    "Misty mountain peaks enveloped in soft clouds, an ancient pine tree rooted on a cliff",
    "A windswept pine leaning over a deep mountain valley at dawn",
    "Three bamboo stalks bending gently beneath spring rain",
    "A flowering plum branch crossing an open sheet of white paper",
    "A solitary lotus rising from a quiet pond with faint ripples",
    "Old willow branches trailing above a narrow riverbank",
    "A cluster of orchids beside a weathered scholar stone",
    "Chrysanthemums blooming beside a simple garden fence",
    "A twisted cypress growing from a rocky island in a calm lake",
    "Wild reeds moving in the wind along an empty shore",
    "A small tea pavilion hidden among pine trees and layered fog",
    "A narrow waterfall descending between distant mountain ridges",
    "A moonlit lake framed by sparse bamboo leaves",
    "A stone footbridge beneath flowering branches in early spring",
    "Terraced mountains fading gradually into pale atmospheric mist",
    "A lone fishing boat crossing a vast river beneath high cliffs",
    "An ancient temple gate surrounded by cedar trees and clouds",
    "A mountain path winding past pines toward a remote hermitage",
    "Two cranes standing among reeds beside still water",
    "A small bird resting on a branch of white plum blossoms",
    "A kingfisher perched above lotus leaves and a quiet stream",
    "Two sparrows sheltering beneath broad banana leaves in rain",
    "A butterfly hovering near a single peony bloom",
    "A heron walking through shallow water beneath hanging willow leaves",
    "A squirrel climbing an old pine branch above an empty valley",
    "A scholar seated beneath a pine, looking toward distant peaks",
    "A tiny traveler crossing a bridge in an immense mountain landscape",
    "A round moon above dark pines and a nearly empty horizon",
    "A weathered bonsai pine in a shallow ceramic pot with ample white space",
    "A close view of ink-washed leaves with one red blossom as the only color",
)


def build_evaluation_records() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "prompt_id": f"eval-{index:02d}",
            "prompt": f"{subject}, {STYLE_TRIGGER}",
            "training_only": False,
            "seeds": [10_001 + index * 10 + offset for offset in range(4)],
        }
        for index, subject in enumerate(EVALUATION_SUBJECTS, start=1)
    )


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description="Create 627 deterministic plant distillation prompts."
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
    parser.add_argument(
        "--prompt-bank",
        type=Path,
        default=root / "data" / "distillation" / "plant_prompt_bank_v1.jsonl",
    )
    parser.add_argument(
        "--evaluation-prompts",
        type=Path,
        default=root / "evaluation" / "distillation_prompts_v1.json",
    )
    parser.add_argument(
        "--output-cache",
        type=Path,
        default=(
            root
            / "data"
            / "features"
            / "distill_t5_plant627_len300_fp16_v1.pt"
        ),
    )
    parser.add_argument("--component-model", default=COMPONENT_MODEL)
    parser.add_argument("--encode-batch-size", type=int, default=1)
    parser.add_argument("--t5-gpu-memory", default="8GiB")
    parser.add_argument("--t5-cpu-memory", default="24GiB")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Write prompt/evaluation manifests without loading T5.",
    )
    return parser


def _load_source_cache(path: Path) -> dict[str, Any]:
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(cache, dict):
        raise ValueError("Source prompt cache must be a dictionary.")
    required = {
        "sample_ids",
        "prompt_embeds",
        "attention_masks",
        "empty_prompt_embeds",
        "empty_prompt_attention_mask",
        "text_encoder_model",
        "manifest_fingerprint",
    }
    missing = sorted(required - set(cache))
    if missing:
        raise ValueError(f"Source prompt cache is missing keys: {missing}")
    if cache["manifest_fingerprint"] != MANIFEST_FINGERPRINT:
        raise ValueError("Source prompt cache fingerprint mismatch.")
    return cache


def _encode_augmented(
    prompts: Sequence[str],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(
        args.component_model,
        subfolder="tokenizer",
        local_files_only=args.local_files_only,
    )
    offload_dir = args.output_cache.parent / "t5_offload_distillation"
    offload_dir.mkdir(parents=True, exist_ok=True)
    encoder = T5EncoderModel.from_pretrained(
        args.component_model,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: args.t5_gpu_memory, "cpu": args.t5_cpu_memory},
        offload_folder=str(offload_dir),
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).eval()
    output_embeddings: list[torch.Tensor] = []
    output_masks: list[torch.Tensor] = []
    for batch_prompts in batched(list(prompts), args.encode_batch_size):
        tokens = tokenizer(
            list(batch_prompts),
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_device = encoder.get_input_embeddings().weight.device
        with torch.inference_mode():
            encoded = encoder(
                input_ids=tokens.input_ids.to(input_device),
                attention_mask=tokens.attention_mask.to(input_device),
            ).last_hidden_state.to("cpu", dtype=torch.float16)
        output_embeddings.append(encoded.contiguous())
        output_masks.append(tokens.attention_mask.to("cpu").contiguous())
    del encoder, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(output_embeddings), torch.cat(output_masks)


def build_assets(args: argparse.Namespace) -> dict[str, Any]:
    if args.encode_batch_size <= 0:
        raise ValueError("--encode-batch-size must be positive.")
    latent_bundle = load_latent_bundle(args.latent_bundle)
    # Older canonical manifests predate the explicit category field. Derive it
    # from the stable sample ID rather than maintaining a runtime patch layer.
    manifest = tuple(
        {
            **row,
            "category": row.get(
                "category", str(row["sample_id"]).split("/", 1)[0]
            ),
        }
        for row in latent_bundle.manifest
    )
    plant_manifest = tuple(
        row for row in manifest if row["category"] == "plant"
    )
    if len(plant_manifest) != 209:
        raise ValueError(f"Expected 209 plant records, got {len(plant_manifest)}.")
    records = build_prompt_records(plant_manifest)
    if len(records) != 627:
        raise RuntimeError(f"Expected 627 prompts, got {len(records)}.")
    bank_fingerprint = save_prompt_bank(args.prompt_bank, records)
    evaluation_records = build_evaluation_records()
    write_json(
        args.evaluation_prompts,
        {
            "format_version": 1,
            "status": "PASS",
            "prompt_bank_fingerprint": bank_fingerprint,
            "training_prompt_ids": [row["prompt_id"] for row in records],
            "prompts": list(evaluation_records),
        },
    )
    result: dict[str, Any] = {
        "status": "TEXT_ONLY" if args.text_only else "PASS",
        "prompt_bank": str(args.prompt_bank.resolve()),
        "prompt_bank_fingerprint": bank_fingerprint,
        "training_prompt_count": len(records),
        "source_sample_count": len(plant_manifest),
        "variants_per_source": 3,
        "evaluation_prompt_count": len(evaluation_records),
        "evaluation_seeds_per_prompt": 4,
        "python": platform.python_version(),
    }
    if args.text_only:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    source = _load_source_cache(args.source_prompt_cache.resolve())
    source_index = {
        sample_id: index for index, sample_id in enumerate(source["sample_ids"])
    }
    missing_source = sorted(
        {row["source_sample_id"] for row in records} - set(source_index)
    )
    if missing_source:
        raise ValueError(
            f"Source prompt cache misses plant IDs: {missing_source[:10]}"
        )

    augmented_records = [row for row in records if row["variant"] != "original"]
    augmented_embeds, augmented_masks = _encode_augmented(
        [row["prompt"] for row in augmented_records], args
    )
    augmented_index = {
        row["prompt_id"]: index for index, row in enumerate(augmented_records)
    }
    embeddings: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for row in records:
        if row["variant"] == "original":
            index = source_index[row["source_sample_id"]]
            embeddings.append(source["prompt_embeds"][index : index + 1])
            masks.append(source["attention_masks"][index : index + 1])
        else:
            index = augmented_index[row["prompt_id"]]
            embeddings.append(augmented_embeds[index : index + 1])
            masks.append(augmented_masks[index : index + 1])
    prompt_embeds = torch.cat(embeddings).contiguous()
    attention_masks = torch.cat(masks).contiguous()
    if prompt_embeds.shape != (627, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
        raise RuntimeError(f"Unexpected embedding shape: {prompt_embeds.shape}")
    if not bool(torch.isfinite(prompt_embeds).all()):
        raise RuntimeError("Generated prompt embeddings are not finite.")
    cache = {
        "format_version": 2,
        "prompt_ids": [row["prompt_id"] for row in records],
        "source_sample_ids": [row["source_sample_id"] for row in records],
        "prompts": [row["prompt"] for row in records],
        "variants": [row["variant"] for row in records],
        "prompt_embeds": prompt_embeds.to(dtype=torch.float16, device="cpu"),
        "attention_masks": attention_masks.to(device="cpu"),
        "empty_prompt_embeds": source["empty_prompt_embeds"].to(
            dtype=torch.float16, device="cpu"
        ),
        "empty_prompt_attention_mask": source[
            "empty_prompt_attention_mask"
        ].to(device="cpu"),
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "text_encoder_model": source["text_encoder_model"],
        "source_manifest_fingerprint": MANIFEST_FINGERPRINT,
        "prompt_bank_fingerprint": bank_fingerprint,
        "prompt_bank_records_fingerprint": fingerprint_records(records),
    }
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output_cache)
    validation = {
        **result,
        "cache_file": args.output_cache.name,
        "prompt_embeds_shape": list(prompt_embeds.shape),
        "prompt_embeds_dtype": str(prompt_embeds.dtype),
        "attention_masks_shape": list(attention_masks.shape),
        "attention_masks_dtype": str(attention_masks.dtype),
        "all_finite": True,
        "text_encoder_model": source["text_encoder_model"],
    }
    write_json(args.output_cache.with_suffix(".validation.json"), validation)
    result["output_cache"] = str(args.output_cache.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_assets(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
