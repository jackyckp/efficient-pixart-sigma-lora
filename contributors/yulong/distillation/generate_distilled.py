#!/usr/bin/env python3
"""Generate a PixArt Teacher or distilled Student from precomputed T5 embeds."""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from diffusers import DDIMScheduler, PixArtSigmaPipeline, PixArtTransformer2DModel
from peft import PeftModel

from common import model_snapshot_source


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSFORMER_MODEL = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
COMPONENT_MODEL = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "precomputed_prompts" / "focused_evaluation_prompts.pt"
DEFAULT_TEACHER = (
    PROJECT_ROOT
    / "models"
    / "lora_training_512"
    / "style_teacher_r16_lr1e-5_bs1_steps10000_seed42"
    / "checkpoints"
    / "step_004000"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "distillation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-file")
    parser.add_argument("--label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument(
        "--memory-mode",
        choices=("cuda", "sequential_cpu_offload"),
        default="cuda",
    )
    return parser.parse_args()


def resolve_adapter(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "adapter_config.json").is_file():
        adapter = path
    elif (path / "lora_adapter" / "adapter_config.json").is_file():
        adapter = path / "lora_adapter"
    else:
        raise FileNotFoundError(f"Invalid LoRA adapter directory: {path}")
    weights = adapter / "adapter_model.safetensors"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty adapter weights: {weights}")
    return adapter


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found.")
    embeddings = args.embeddings.expanduser().resolve()
    if not embeddings.is_file():
        raise FileNotFoundError(f"Missing embedding bundle: {embeddings}")
    if args.steps < 1 or args.guidance < 1.0:
        raise ValueError("--steps must be positive and --guidance must be >= 1")
    adapter = None if args.base else resolve_adapter(args.adapter)

    bundle = torch.load(embeddings, map_location="cpu", weights_only=True)
    if bundle.get("format") != "pixart_sigma_precomputed_prompt_embeddings_v1":
        raise ValueError("This is not a compatible precomputed prompt bundle.")
    prompts = bundle["prompts"]
    if not 0 <= args.prompt_index < len(prompts):
        raise IndexError(f"--prompt-index must be 0..{len(prompts) - 1}")
    prompt = prompts[args.prompt_index]
    prompt_embeds = bundle["prompt_embeds"][args.prompt_index : args.prompt_index + 1].to(
        "cuda", dtype=torch.float16
    )
    prompt_mask = bundle["prompt_attention_masks"][
        args.prompt_index : args.prompt_index + 1
    ].to("cuda")
    empty_embeds = bundle["empty_prompt_embeds"].to("cuda", dtype=torch.float16)
    empty_mask = bundle["empty_prompt_attention_mask"].to("cuda")

    torch.backends.cuda.matmul.allow_tf32 = True
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("Loading PixArt transformer; T5 remains omitted.")
    transformer = PixArtTransformer2DModel.from_pretrained(
        model_snapshot_source(TRANSFORMER_MODEL),
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    if adapter is None:
        model_label = args.label or "base"
        transformer = transformer.eval()
    else:
        model_label = args.label or adapter.parent.name
        transformer = (
            PeftModel.from_pretrained(
                transformer, adapter, is_trainable=False, low_cpu_mem_usage=True
            )
            .eval()
            .merge_and_unload(safe_merge=True)
            .eval()
        )

    scheduler = DDIMScheduler.from_pretrained(
        model_snapshot_source(COMPONENT_MODEL),
        subfolder="scheduler",
        timestep_spacing="trailing",
        clip_sample=False,
    )
    pipe = PixArtSigmaPipeline.from_pretrained(
        model_snapshot_source(COMPONENT_MODEL),
        transformer=transformer,
        scheduler=scheduler,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    if args.memory_mode == "cuda":
        pipe.to("cuda")
    else:
        pipe.enable_sequential_cpu_offload()
    pipe.set_progress_bar_config(disable=False)

    call_args = {
        "prompt": None,
        "prompt_embeds": prompt_embeds,
        "prompt_attention_mask": prompt_mask,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "height": 512,
        "width": 512,
        "generator": torch.Generator(device=pipe._execution_device).manual_seed(
            args.seed
        ),
        "use_resolution_binning": False,
    }
    if args.guidance > 1.0:
        call_args["negative_prompt"] = None
        call_args["negative_prompt_embeds"] = empty_embeds
        call_args["negative_prompt_attention_mask"] = empty_mask

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_file and Path(args.output_file).name != args.output_file:
        raise ValueError("--output-file must be a filename, not a path")
    started = time.perf_counter()
    with torch.inference_mode():
        image = pipe(**call_args).images[0]
    torch.cuda.synchronize()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = args.output_file or (
        f"{stamp}_{model_label}_p{args.prompt_index:02d}_"
        f"s{args.steps}_g{args.guidance:g}_seed{args.seed}.png"
    )
    image_path = output_dir / filename
    image.save(image_path)
    record = {
        "image_file": image_path.name,
        "model_label": model_label,
        "prompt": prompt,
        "prompt_index": args.prompt_index,
        "seed": args.seed,
        "steps": args.steps,
        "guidance": args.guidance,
        "adapter": None if adapter is None else str(adapter),
        "scheduler": "DDIMScheduler",
        "timestep_spacing": "trailing",
        "memory_mode": args.memory_mode,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    with (output_dir / "generation_metadata.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved   : {image_path}")
    print(f"Prompt  : {prompt}")
    print(f"Schedule: DDIM trailing, {args.steps} steps, guidance {args.guidance:g}")
    print(f"Elapsed : {record['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
