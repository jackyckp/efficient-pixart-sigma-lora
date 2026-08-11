"""Train a PixArt-Sigma PEFT LoRA locally from precomputed caches.

The defaults reproduce the planned local experiment:
  * RTX GPU, fp16, batch size 1, gradient accumulation 1
  * LoRA rank 16, alpha 16
  * learning rate 1e-6
  * 10000 optimizer updates
  * adapter-only checkpoints every 1000 optimizer updates

The script intentionally does not load the VAE or T5 encoder. Image latents and
T5 prompt embeddings must already have been computed by the companion Colab
notebooks and copied into the local training_data directory.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TRANSFORMER_MODEL = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
COMPONENT_MODEL = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
DEFAULT_LATENT_DIR = PROJECT_ROOT / "training_data" / "clean_latents_512"
DEFAULT_TEXT_DIR = PROJECT_ROOT / "training_data" / "t5_embeddings_512"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "models" / "lora_training_512"

OFFICIAL_TARGET_MODULES = [
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "proj_in",
    "proj_out",
    "ff.net.0.proj",
    "ff.net.2",
    "proj",
    "linear",
    "linear_1",
    "linear_2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-cache-dir", type=Path, default=DEFAULT_LATENT_DIR)
    parser.add_argument("--text-cache-dir", type=Path, default=DEFAULT_TEXT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", help="Optional output folder name.")
    parser.add_argument("--max-train-steps", type=int, default=10000)
    parser.add_argument("--checkpointing-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--caption-dropout", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-train-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate CUDA, packages, and both cache files without loading the model.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run exactly one optimizer update and save a separate smoke-test adapter.",
    )
    return parser.parse_args()


def resolve_single_cache(directory: Path, pattern: str) -> Path:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Cache directory does not exist: {directory}\n"
            "Copy the matching .pt cache from Google Drive into this directory."
        )
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No file matching {pattern!r} was found in {directory}")
    if len(candidates) > 1:
        names = "\n".join(f"- {path.name}" for path in candidates)
        raise RuntimeError(
            f"Expected exactly one cache in {directory}, found {len(candidates)}:\n"
            f"{names}\nMove obsolete cache files out of this directory."
        )
    return candidates[0]


def load_and_validate_caches(args: argparse.Namespace, torch):
    latent_cache_path = resolve_single_cache(
        args.latent_cache_dir, "image_latents_n*_res512_*.pt"
    )
    text_cache_path = resolve_single_cache(
        args.text_cache_dir, "t5_embeddings_n*_len300_fp16_*.pt"
    )
    image_cache = torch.load(latent_cache_path, map_location="cpu", weights_only=True)
    text_cache = torch.load(text_cache_path, map_location="cpu", weights_only=True)

    required_image_keys = {
        "latents",
        "sample_ids",
        "manifest_fingerprint",
        "transformer_model",
        "resolution",
        "latent_kind",
    }
    required_text_keys = {
        "prompt_embeds",
        "attention_masks",
        "sample_ids",
        "manifest_fingerprint",
        "transformer_model",
        "max_sequence_length",
        "empty_prompt_embeds",
        "empty_prompt_attention_mask",
    }
    missing_image = required_image_keys - set(image_cache)
    missing_text = required_text_keys - set(text_cache)
    if missing_image:
        raise KeyError(f"Image cache is missing keys: {sorted(missing_image)}")
    if missing_text:
        raise KeyError(f"Text cache is missing keys: {sorted(missing_text)}")

    clean_latents = image_cache["latents"]
    prompt_embeds = text_cache["prompt_embeds"]
    attention_masks = text_cache["attention_masks"]
    empty_prompt_embeds = text_cache["empty_prompt_embeds"]
    empty_prompt_attention_mask = text_cache["empty_prompt_attention_mask"]
    sample_ids = image_cache["sample_ids"]

    if sample_ids != text_cache["sample_ids"]:
        raise ValueError("Image and text sample_ids differ; conditions are misaligned.")
    if image_cache["manifest_fingerprint"] != text_cache["manifest_fingerprint"]:
        raise ValueError("Image and text caches were built from different manifests.")
    if image_cache["transformer_model"] != TRANSFORMER_MODEL:
        raise ValueError("Image cache transformer model does not match this script.")
    if text_cache["transformer_model"] != TRANSFORMER_MODEL:
        raise ValueError("Text cache transformer model does not match this script.")
    if image_cache["resolution"] != 512:
        raise ValueError(f"Expected resolution 512, got {image_cache['resolution']}")
    if image_cache["latent_kind"] != "clean_x0_scaled":
        raise ValueError(f"Unexpected latent kind: {image_cache['latent_kind']}")
    if text_cache["max_sequence_length"] != 300:
        raise ValueError("Expected max_sequence_length=300.")

    num_aligned = len(sample_ids)
    expected_shapes = {
        "latents": ((num_aligned, 4, 64, 64), clean_latents.shape),
        "prompt_embeds": ((num_aligned, 300, 4096), prompt_embeds.shape),
        "attention_masks": ((num_aligned, 300), attention_masks.shape),
        "empty_prompt_embeds": ((1, 300, 4096), empty_prompt_embeds.shape),
        "empty_prompt_attention_mask": (
            (1, 300),
            empty_prompt_attention_mask.shape,
        ),
    }
    for name, (expected, actual) in expected_shapes.items():
        if tuple(actual) != expected:
            raise ValueError(f"{name} shape is {tuple(actual)}, expected {expected}")
    if clean_latents.dtype != torch.float16:
        raise TypeError(f"Latents must be float16, got {clean_latents.dtype}")
    if prompt_embeds.dtype != torch.float16:
        raise TypeError(f"Prompt embeddings must be float16, got {prompt_embeds.dtype}")
    if attention_masks.dtype != torch.int64:
        raise TypeError(f"Attention masks must be int64, got {attention_masks.dtype}")
    if not torch.isfinite(clean_latents).all():
        raise ValueError("Latent cache contains NaN or Inf.")
    if not torch.isfinite(prompt_embeds).all():
        raise ValueError("Text cache contains NaN or Inf.")
    if not set(torch.unique(attention_masks).tolist()).issubset({0, 1}):
        raise ValueError("Attention masks contain values other than 0 and 1.")
    if len(set(sample_ids)) != num_aligned:
        raise ValueError("Duplicate sample_ids were found.")

    num_train = num_aligned if args.num_train_samples is None else args.num_train_samples
    if not 1 <= num_train <= num_aligned:
        raise ValueError(
            f"--num-train-samples must be 1..{num_aligned}, got {num_train}"
        )

    result = {
        "latent_cache_path": latent_cache_path,
        "text_cache_path": text_cache_path,
        "image_cache": image_cache,
        "clean_latents": clean_latents[:num_train].contiguous(),
        "prompt_embeds": prompt_embeds[:num_train].contiguous(),
        "attention_masks": attention_masks[:num_train].contiguous(),
        "empty_prompt_embeds": empty_prompt_embeds,
        "empty_prompt_attention_mask": empty_prompt_attention_mask,
        "sample_ids": sample_ids[:num_train],
        "num_aligned": num_aligned,
        "num_train": num_train,
    }
    print("CACHE VALIDATION PASSED")
    print(f"Latent cache     : {latent_cache_path}")
    print(f"Text cache       : {text_cache_path}")
    print(f"Aligned samples  : {num_aligned}")
    print(f"Training samples : {num_train}")
    print(f"Fingerprint      : {image_cache['manifest_fingerprint']}")
    return result


def learning_rate_tag(value: float) -> str:
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def build_run_dir(args: argparse.Namespace, num_train: int) -> Path:
    steps = 1 if args.smoke_test else args.max_train_steps
    subset_tag = "nall" if args.num_train_samples is None else f"n{num_train}"
    default_name = (
        f"pixart_sigma_512_lora_r{args.rank}_{subset_tag}_"
        f"bs{args.batch_size}_ga{args.gradient_accumulation_steps}_"
        f"lr{learning_rate_tag(args.learning_rate)}_steps{steps}_seed{args.seed}"
    )
    if args.smoke_test:
        default_name += "_smoke"
    run_dir = (args.output_root / (args.run_name or default_name)).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {run_dir}\n"
            "Use --run-name with a new experiment name so existing results are preserved."
        )
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> int:
    args = parse_args()
    if args.max_train_steps < 1:
        raise ValueError("--max-train-steps must be at least 1")
    if args.checkpointing_steps < 1:
        raise ValueError("--checkpointing-steps must be at least 1")
    if args.batch_size != 1:
        print("WARNING: batch sizes above 1 are likely to exceed an 8 GB GPU.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import DDPMScheduler, PixArtTransformer2DModel
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Install the CUDA build of PyTorch.")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Python  : {platform.python_version()}")
    print(f"PyTorch : {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"GPU     : {gpu_name} ({vram_gb:.1f} GiB)")
    if vram_gb < 7.5:
        print("WARNING: less than 7.5 GiB of CUDA memory is visible.")

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    cache = load_and_validate_caches(args, torch)
    if args.check_only:
        print("CHECK-ONLY PASSED: no model was loaded and no training was started.")
        return 0

    run_dir = build_run_dir(args, cache["num_train"])
    checkpoint_dir = run_dir / "checkpoints"
    final_adapter_dir = run_dir / "final_adapter"
    max_train_steps = 1 if args.smoke_test else args.max_train_steps
    checkpointing_steps = 1 if args.smoke_test else args.checkpointing_steps
    print(f"Run output: {run_dir}")

    class CachedPixArtDataset(Dataset):
        def __len__(self):
            return len(cache["sample_ids"])

        def __getitem__(self, index):
            return {
                "latents": cache["clean_latents"][index],
                "prompt_embeds": cache["prompt_embeds"][index],
                "attention_mask": cache["attention_masks"][index],
                "sample_id": cache["sample_ids"][index],
            }

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        CachedPixArtDataset(),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=loader_generator,
    )

    gc.collect()
    torch.cuda.empty_cache()
    noise_scheduler = DDPMScheduler.from_pretrained(
        COMPONENT_MODEL, subfolder="scheduler"
    )
    print("Loading PixArt-Sigma transformer (the first run may download it)...")
    transformer = PixArtTransformer2DModel.from_pretrained(
        TRANSFORMER_MODEL,
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        init_lora_weights="gaussian",
        target_modules=OFFICIAL_TARGET_MODULES,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    for parameter in transformer.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)
    trainable = [
        (name, parameter)
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or not all("lora_" in name for name, _ in trainable):
        raise RuntimeError("LoRA attachment validation failed.")
    transformer.print_trainable_parameters()

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps,
    )
    transformer, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_loader, lr_scheduler
    )
    empty_prompt_embeds_gpu = cache["empty_prompt_embeds"].to(
        accelerator.device, dtype=torch.float16
    )
    empty_prompt_attention_mask_gpu = cache["empty_prompt_attention_mask"].to(
        accelerator.device
    )
    effective_batch_size = (
        args.batch_size
        * args.gradient_accumulation_steps
        * accelerator.num_processes
    )
    print("Optimizer: torch.optim.AdamW (Windows-compatible)")
    print(f"Effective batch size per optimizer update: {effective_batch_size}")
    print(f"Optimizer updates: {max_train_steps}")

    def save_adapter_checkpoint(step: int, losses: list[float], peak_vram: float):
        checkpoint_path = checkpoint_dir / f"step_{step:06d}"
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(transformer)
            unwrapped.save_pretrained(checkpoint_path, safe_serialization=True)
            metadata = {
                "optimizer_step": step,
                "loss_history": losses,
                "peak_allocated_vram_gb": peak_vram,
                "note": "Adapter-only checkpoint; optimizer state is not included.",
            }
            (checkpoint_path / "checkpoint_metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
        return checkpoint_path

    global_step = 0
    micro_step = 0
    loss_history: list[float] = []
    learning_rate_history: list[float] = []
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    transformer.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        total=max_train_steps,
        desc="Optimizer updates",
        disable=not accelerator.is_local_main_process,
    )

    while global_step < max_train_steps:
        for batch in train_loader:
            with accelerator.accumulate(transformer):
                latents = batch["latents"].to(
                    accelerator.device, dtype=torch.float16, non_blocking=True
                )
                batch_prompt_embeds = batch["prompt_embeds"].to(
                    accelerator.device, dtype=torch.float16, non_blocking=True
                )
                batch_attention_mask = batch["attention_mask"].to(
                    accelerator.device, non_blocking=True
                )
                batch_size = latents.shape[0]
                if args.caption_dropout > 0:
                    drop = (
                        torch.rand(batch_size, device=latents.device)
                        < args.caption_dropout
                    )
                    if drop.any():
                        batch_prompt_embeds = batch_prompt_embeds.clone()
                        batch_attention_mask = batch_attention_mask.clone()
                        batch_prompt_embeds[drop] = empty_prompt_embeds_gpu.expand(
                            batch_size, -1, -1
                        )[drop]
                        batch_attention_mask[drop] = (
                            empty_prompt_attention_mask_gpu.expand(batch_size, -1)[drop]
                        )

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                prediction_type = noise_scheduler.config.prediction_type
                if prediction_type == "epsilon":
                    target = noise
                elif prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction type: {prediction_type}")

                with accelerator.autocast():
                    model_output = transformer(
                        noisy_latents,
                        encoder_hidden_states=batch_prompt_embeds,
                        encoder_attention_mask=batch_attention_mask,
                        timestep=timesteps,
                        added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                    ).sample
                    model_prediction = model_output.chunk(2, dim=1)[0]
                    loss = F.mse_loss(
                        model_prediction.float(), target.float(), reduction="mean"
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss before optimizer step {global_step + 1}: "
                        f"{loss.item()}"
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        transformer.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            micro_step += 1
            if accelerator.sync_gradients:
                global_step += 1
                gathered_loss = accelerator.gather(loss.detach().reshape(1)).mean().item()
                current_lr = lr_scheduler.get_last_lr()[0]
                loss_history.append(gathered_loss)
                learning_rate_history.append(current_lr)
                progress.update(1)
                progress.set_postfix(loss=f"{gathered_loss:.5f}", lr=f"{current_lr:.2e}")
                if (
                    global_step % checkpointing_steps == 0
                    or global_step == max_train_steps
                ):
                    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
                    path = save_adapter_checkpoint(global_step, loss_history, peak_vram)
                    print(f"\nSaved adapter checkpoint: {path}")
            if global_step >= max_train_steps:
                break

    progress.close()
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    if global_step != max_train_steps:
        raise RuntimeError(f"Stopped at {global_step}, expected {max_train_steps}")
    if len(loss_history) != global_step or not all(map(math.isfinite, loss_history)):
        raise RuntimeError("Loss-history validation failed.")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_model = accelerator.unwrap_model(transformer)
        final_model.save_pretrained(final_adapter_dir, safe_serialization=True)
        run_metadata = {
            "status": "PASS",
            "transformer_model": TRANSFORMER_MODEL,
            "component_model": COMPONENT_MODEL,
            "latent_cache_file": cache["latent_cache_path"].name,
            "text_cache_file": cache["text_cache_path"].name,
            "manifest_fingerprint": cache["image_cache"]["manifest_fingerprint"],
            "num_aligned_samples": cache["num_aligned"],
            "num_train_samples": cache["num_train"],
            "resolution": 512,
            "rank": args.rank,
            "lora_alpha": args.alpha,
            "optimizer": "torch.optim.AdamW",
            "optimizer_steps": global_step,
            "micro_steps": micro_step,
            "train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "caption_dropout": args.caption_dropout,
            "seed": args.seed,
            "loss_history": loss_history,
            "learning_rate_history": learning_rate_history,
            "train_seconds": train_seconds,
            "peak_allocated_vram_gb": peak_vram_gb,
            "gpu": gpu_name,
            "python": platform.python_version(),
            "torch": torch.__version__,
        }
        (run_dir / "training_metadata.json").write_text(
            json.dumps(run_metadata, indent=2), encoding="utf-8"
        )
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(loss_history) + 1), loss_history, linewidth=1.0)
        plt.xlabel("Optimizer step")
        plt.ylabel("Diffusion MSE loss")
        plt.title("PixArt-Sigma LoRA Training Loss")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(run_dir / "training_loss.png", dpi=160)
        plt.close()

    adapter_config = final_adapter_dir / "adapter_config.json"
    adapter_weights = final_adapter_dir / "adapter_model.safetensors"
    if not adapter_config.is_file() or not adapter_weights.is_file():
        raise FileNotFoundError("Final adapter files were not saved correctly.")
    print(
        f"TRAINING COMPLETE: {global_step} optimizer updates in "
        f"{train_seconds / 60:.1f} minutes; peak VRAM={peak_vram_gb:.2f} GiB"
    )
    print(f"Final adapter: {final_adapter_dir}")
    print(f"Loss curve  : {run_dir / 'training_loss.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
