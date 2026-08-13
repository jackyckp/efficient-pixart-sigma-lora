#!/usr/bin/env python3
"""Generate with a joint PixArt distilled LoRA in exactly 2 or 4 calls."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path
from typing import Sequence
import sys

import torch

from scripts.distillation.common import (
    COMPONENT_MODEL,
    LATENT_SHAPE,
    MAX_SEQUENCE_LENGTH,
    TRANSFORMER_MODEL,
    deterministic_jump,
    load_distill_prompt_cache,
    phase_pairs,
    repository_root,
    resolve_adapter_dir,
    split_epsilon_prediction,
    state_timestep,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exact-call-count PixArt phased LoRA inference."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompt-id")
    parser.add_argument("--prompt-cache", type=Path, default=None)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-inference-steps", type=int, choices=(2, 4), required=True)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transformer-model", default=TRANSFORMER_MODEL)
    parser.add_argument("--component-model", default=COMPONENT_MODEL)
    parser.add_argument("--t5-gpu-memory", default="8GiB")
    parser.add_argument("--t5-cpu-memory", default="24GiB")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _encode_prompt(
    prompt: str,
    args: argparse.Namespace,
    offload_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(
        args.component_model,
        subfolder="tokenizer",
        local_files_only=args.local_files_only,
    )
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
    tokens = tokenizer(
        [prompt],
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_device = encoder.get_input_embeddings().weight.device
    with torch.inference_mode():
        embeds = encoder(
            input_ids=tokens.input_ids.to(input_device),
            attention_mask=tokens.attention_mask.to(input_device),
        ).last_hidden_state.to("cpu", dtype=torch.float16)
    mask = tokens.attention_mask.to("cpu")
    del encoder, tokenizer, tokens
    gc.collect()
    torch.cuda.empty_cache()
    return embeds, mask


def _load_prompt_features(
    args: argparse.Namespace, output_dir: Path
) -> tuple[str, torch.Tensor, torch.Tensor, str | None]:
    if args.prompt is not None:
        prompt = " ".join(args.prompt.split())
        if not prompt:
            raise ValueError("--prompt may not be empty.")
        embeds, mask = _encode_prompt(
            prompt, args, output_dir / "t5_offload_distilled"
        )
        return prompt, embeds, mask, None
    if args.prompt_cache is None:
        raise ValueError("--prompt-id requires --prompt-cache.")
    features = load_distill_prompt_cache(args.prompt_cache)
    try:
        index = features.prompt_ids.index(args.prompt_id)
    except ValueError as error:
        raise ValueError(f"Prompt ID not found: {args.prompt_id}") from error
    cache = torch.load(
        features.path, map_location="cpu", weights_only=True
    )
    prompt_text = cache.get("prompts", [args.prompt_id] * len(features.prompt_ids))[
        index
    ]
    return (
        prompt_text,
        features.prompt_embeds[index : index + 1],
        features.attention_masks[index : index + 1],
        args.prompt_id,
    )


def _declared_target_steps(adapter_dir: Path) -> int | None:
    candidates = (
        adapter_dir.parent / "checkpoint_metadata.json",
        adapter_dir.parent / "run_metadata.json",
        adapter_dir.parent.parent / "run_metadata.json",
    )
    for path in candidates:
        if path.is_file():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            value = metadata.get("target_inference_steps")
            if value is not None:
                return int(value)
    return None


def generate(args: argparse.Namespace) -> dict[str, object]:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"Expected Python 3.11.x, got {platform.python_version()}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")
    if args.guidance_scale != 1.0:
        raise ValueError(
            "Distilled inference fixes guidance_scale=1.0 and has no CFG branch."
        )
    adapter_dir = resolve_adapter_dir(args.adapter)
    declared_steps = _declared_target_steps(adapter_dir)
    if declared_steps is not None and declared_steps != args.num_inference_steps:
        raise ValueError(
            f"Adapter declares {declared_steps} steps, but CLI requested "
            f"{args.num_inference_steps}."
        )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt, prompt_embed, prompt_mask, prompt_id = _load_prompt_features(
        args, output_path.parent
    )

    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        PixArtTransformer2DModel,
    )
    from diffusers.image_processor import VaeImageProcessor
    from peft import PeftModel

    device = torch.device("cuda")
    base = PixArtTransformer2DModel.from_pretrained(
        args.transformer_model,
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    )
    transformer = PeftModel.from_pretrained(
        base, adapter_dir, is_trainable=False
    ).to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(
        args.component_model,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )
    alphas = scheduler.alphas_cumprod.to(torch.float32)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn(
        (1, *LATENT_SHAPE),
        generator=generator,
        device=device,
        dtype=torch.float16,
    )
    prompt_embed = prompt_embed.to(device)
    prompt_mask = prompt_mask.to(device)
    call_count = 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for start_index, target_index in phase_pairs(args.num_inference_steps):
            start_t = state_timestep(start_index)
            target_t = state_timestep(target_index)
            timesteps = torch.full((1,), start_t, device=device, dtype=torch.long)
            output = transformer(
                latents,
                encoder_hidden_states=prompt_embed,
                encoder_attention_mask=prompt_mask,
                timestep=timesteps,
                added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                return_dict=False,
            )[0]
            call_count += 1
            epsilon = split_epsilon_prediction(output)
            latents = deterministic_jump(
                latents, epsilon, start_t, target_t, alphas
            ).to(dtype=torch.float16)
    torch.cuda.synchronize()
    denoise_seconds = time.perf_counter() - started
    if call_count != args.num_inference_steps:
        raise RuntimeError(
            f"Expected {args.num_inference_steps} transformer calls, got {call_count}."
        )
    if not bool(torch.isfinite(latents).all()):
        raise FloatingPointError("Distilled final latent is not finite.")

    del transformer, base
    gc.collect()
    torch.cuda.empty_cache()
    vae = AutoencoderKL.from_pretrained(
        args.component_model,
        subfolder="vae",
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    with torch.inference_mode():
        decoded = vae.decode(
            latents / vae.config.scaling_factor,
            return_dict=False,
        )[0]
    image = VaeImageProcessor(vae_scale_factor=8).postprocess(
        decoded, output_type="pil"
    )[0]
    image.save(output_path)
    result: dict[str, object] = {
        "status": "PASS",
        "prompt": prompt,
        "prompt_id": prompt_id,
        "adapter": str(adapter_dir),
        "adapter_rank": 16,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": 1.0,
        "classifier_free_guidance_branch": False,
        "transformer_forward_calls": call_count,
        "phase_index_pairs": [
            list(pair) for pair in phase_pairs(args.num_inference_steps)
        ],
        "denoise_seconds": denoise_seconds,
        "peak_allocated_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "image": str(output_path),
        "image_size": list(image.size),
        "transformer_model": args.transformer_model,
        "component_model": args.component_model,
    }
    write_json(output_path.with_suffix(".json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
