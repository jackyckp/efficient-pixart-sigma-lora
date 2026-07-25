#!/usr/bin/env python3
"""Generate a PixArt-Sigma LoRA image from an arbitrary unseen prompt."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

import torch


TRANSFORMER_MODEL = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
COMPONENT_MODEL = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
MAX_SEQUENCE_LENGTH = 300


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description=(
            "Encode a new text prompt with T5, release T5, then generate "
            "with the local PixArt-Sigma LoRA adapter."
        )
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=(
            root
            / "outputs"
            / "local_smoke"
            / "r8_n50_steps100"
            / "lora_adapter"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs" / "unseen_prompt.png",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--t5-gpu-memory", default="4GiB")
    parser.add_argument("--t5-cpu-memory", default="8GiB")
    parser.add_argument("--allow-seen-prompt", action="store_true")
    parser.add_argument("--transformer-model", default=TRANSFORMER_MODEL)
    parser.add_argument("--component-model", default=COMPONENT_MODEL)
    return parser


def audit_unseen_prompt(
    prompt: str,
    adapter_dir: Path,
    allow_seen_prompt: bool,
) -> dict[str, object]:
    subset_path = adapter_dir.parent / "subset_manifest.json"
    audit: dict[str, object] = {
        "subset_manifest": str(subset_path),
        "subset_manifest_found": subset_path.is_file(),
        "exact_training_caption_match": False,
    }
    if not subset_path.is_file():
        return audit
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    normalized_prompt = " ".join(prompt.split()).casefold()
    exact_matches = [
        row["sample_id"]
        for row in subset
        if " ".join(row["caption"].split()).casefold()
        == normalized_prompt
    ]
    audit["training_subset_size"] = len(subset)
    audit["exact_training_caption_match"] = bool(exact_matches)
    audit["matching_sample_ids"] = exact_matches
    if exact_matches and not allow_seen_prompt:
        raise ValueError(
            "Prompt exactly matches a training caption for IDs "
            f"{exact_matches}. Pass --allow-seen-prompt to override."
        )
    return audit


def encode_prompts(
    args: argparse.Namespace,
    offload_dir: Path,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    float,
]:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(
        args.component_model,
        subfolder="tokenizer",
    )
    offload_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    text_encoder = T5EncoderModel.from_pretrained(
        args.component_model,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={
            0: args.t5_gpu_memory,
            "cpu": args.t5_cpu_memory,
        },
        offload_folder=str(offload_dir),
        offload_state_dict=True,
        low_cpu_mem_usage=True,
    ).eval()

    texts = [args.prompt]
    if args.guidance_scale > 1.0:
        texts.append(args.negative_prompt)
    tokens = tokenizer(
        texts,
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_device = text_encoder.get_input_embeddings().weight.device
    with torch.inference_mode():
        embeddings = text_encoder(
            input_ids=tokens.input_ids.to(input_device),
            attention_mask=tokens.attention_mask.to(input_device),
        ).last_hidden_state.to("cpu", dtype=torch.float16)
    masks = tokens.attention_mask.to("cpu")
    encode_seconds = time.perf_counter() - start

    prompt_embeds = embeddings[:1].contiguous()
    prompt_mask = masks[:1].contiguous()
    negative_embeds = None
    negative_mask = None
    if args.guidance_scale > 1.0:
        negative_embeds = embeddings[1:2].contiguous()
        negative_mask = masks[1:2].contiguous()

    del embeddings, masks, tokens, text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return (
        prompt_embeds,
        prompt_mask,
        negative_embeds,
        negative_mask,
        encode_seconds,
    )


def generate(args: argparse.Namespace) -> dict[str, object]:
    if sys.version_info[:3] != (3, 11, 2):
        raise RuntimeError(
            f"Expected Python 3.11.2, got {platform.python_version()}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")
    if not args.prompt.strip():
        raise ValueError("--prompt may not be empty.")
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive.")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 1.0:
        raise ValueError("--guidance-scale must be finite and at least 1.0.")

    from diffusers import PixArtSigmaPipeline, PixArtTransformer2DModel
    from peft import PeftModel

    adapter_dir = args.adapter.resolve()
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"PEFT adapter not found: {adapter_dir}")
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_audit = audit_unseen_prompt(
        args.prompt,
        adapter_dir,
        args.allow_seen_prompt,
    )

    offload_dir = output_path.parent / "t5_offload"
    (
        prompt_embeds,
        prompt_mask,
        negative_embeds,
        negative_mask,
        encode_seconds,
    ) = encode_prompts(args, offload_dir)

    base_transformer = PixArtTransformer2DModel.from_pretrained(
        args.transformer_model,
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    reloaded_transformer = PeftModel.from_pretrained(
        base_transformer,
        adapter_dir,
        is_trainable=False,
    ).eval()
    loaded_rank = reloaded_transformer.peft_config["default"].r
    transformer = reloaded_transformer.get_base_model().eval()
    pipe = PixArtSigmaPipeline.from_pretrained(
        args.component_model,
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")

    generator = torch.Generator("cuda").manual_seed(args.seed)
    generation_kwargs = {
        "prompt": None,
        "prompt_embeds": prompt_embeds.to("cuda"),
        "prompt_attention_mask": prompt_mask.to("cuda"),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": 512,
        "width": 512,
        "use_resolution_binning": False,
        "generator": generator,
    }
    if negative_embeds is not None and negative_mask is not None:
        generation_kwargs.update(
            negative_prompt_embeds=negative_embeds.to("cuda"),
            negative_prompt_attention_mask=negative_mask.to("cuda"),
        )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        image = pipe(**generation_kwargs).images[0]
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - start
    image.save(output_path)

    metadata = {
        "status": "PASS",
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "prompt_audit": prompt_audit,
        "adapter": str(adapter_dir),
        "adapter_rank": loaded_rank,
        "transformer_model": args.transformer_model,
        "component_model": args.component_model,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "t5_encode_seconds": encode_seconds,
        "generation_seconds": generation_seconds,
        "peak_generation_vram_gb": (
            torch.cuda.max_memory_allocated() / 1024**3
        ),
        "image": str(output_path),
        "image_size": list(image.size),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
