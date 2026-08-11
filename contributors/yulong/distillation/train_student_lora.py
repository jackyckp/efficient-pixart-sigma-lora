#!/usr/bin/env python3
"""Train a PixArt-Sigma Student LoRA from cached progressive targets."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from common import (
    COMPONENT_MODEL,
    TRANSFORMER_MODEL,
    file_sha256,
    load_training_caches,
    load_transformer_with_adapter,
    model_epsilon,
    resolve_adapter,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LATENTS = (
    PROJECT_ROOT
    / "training_data"
    / "clean_latents_512"
    / "image_latents_n260_res512_b9d3c2d1d404.pt"
)
DEFAULT_TEXT = (
    PROJECT_ROOT
    / "training_data"
    / "t5_embeddings_512"
    / "t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt"
)
DEFAULT_INIT_ADAPTER = (
    PROJECT_ROOT
    / "models"
    / "lora_training_512"
    / "style_teacher_r16_lr1e-5_bs1_steps10000_seed42"
    / "checkpoints"
    / "step_004000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path, default=DEFAULT_INIT_ADAPTER)
    parser.add_argument("--latent-cache", type=Path, default=DEFAULT_LATENTS)
    parser.add_argument("--text-cache", type=Path, default=DEFAULT_TEXT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "models" / "distilled_students",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--max-train-steps", type=int, default=4000)
    parser.add_argument("--checkpointing-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run exactly one optimizer update and save a smoke adapter.",
    )
    return parser.parse_args()


def load_targets(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing target cache: {path}")
    targets = torch.load(path, map_location="cpu", weights_only=True)
    supported_formats = {
        "pixart_progressive_distillation_targets_v1",
        "pixart_progressive_distillation_targets_v2_teacher_trajectory",
    }
    if targets.get("format") not in supported_formats:
        raise ValueError(f"Unsupported target cache format in {path}")
    required = {
        "noisy_latents",
        "target_epsilons",
        "sample_indices",
        "timesteps",
        "metadata",
    }
    if missing := required.difference(targets):
        raise KeyError(f"Target cache is missing keys: {sorted(missing)}")
    noisy = targets["noisy_latents"]
    epsilon = targets["target_epsilons"]
    count = noisy.shape[0]
    if tuple(noisy.shape[1:]) != (4, 64, 64):
        raise ValueError(f"Unexpected noisy latent shape: {tuple(noisy.shape)}")
    if noisy.shape != epsilon.shape:
        raise ValueError("Noisy latent and target epsilon shapes differ.")
    if noisy.dtype != torch.float16 or epsilon.dtype != torch.float16:
        raise TypeError("Distillation target tensors must be float16.")
    if targets["sample_indices"].shape != (count,):
        raise ValueError("sample_indices length does not match target records.")
    if targets["timesteps"].shape != (count,):
        raise ValueError("timesteps length does not match target records.")
    if not torch.isfinite(noisy).all() or not torch.isfinite(epsilon).all():
        raise FloatingPointError("Target cache contains NaN or Inf.")
    targets["path"] = path
    return targets


def build_run_dir(args: argparse.Namespace, metadata: dict) -> Path:
    source_steps = metadata["source_steps"]
    student_steps = metadata["student_steps"]
    max_steps = 1 if args.smoke_test else args.max_train_steps
    default_name = (
        f"pixart_student_{source_steps}to{student_steps}_"
        f"lora_lr{args.learning_rate:.0e}_train{max_steps}_seed{args.seed}"
    ).replace("e-0", "e-")
    if args.smoke_test:
        default_name += "_smoke"
    run_dir = (args.output_root / (args.run_name or default_name)).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {run_dir}\n"
            "Use --run-name with a new experiment name."
        )
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> int:
    args = parse_args()
    if args.max_train_steps < 1 or args.checkpointing_steps < 1:
        raise ValueError("Training and checkpoint step counts must be positive.")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Batch size and accumulation must be positive.")
    if args.batch_size > 1:
        print("WARNING: batch size above 1 may exceed an 8 GiB GPU.")

    init_adapter = resolve_adapter(args.init_adapter)
    targets = load_targets(args.target_cache)
    training_cache = load_training_caches(args.latent_cache, args.text_cache)
    target_metadata = targets["metadata"]
    if target_metadata["manifest_fingerprint"] != training_cache["fingerprint"]:
        raise ValueError("Target cache and text cache fingerprints do not match.")
    if targets["sample_indices"].max().item() >= len(training_cache["sample_ids"]):
        raise IndexError("Target cache references a missing prompt sample.")
    init_hash = file_sha256(init_adapter / "adapter_model.safetensors")
    expected_hash = target_metadata["teacher_adapter_sha256"]
    if init_hash != expected_hash:
        raise ValueError(
            "--init-adapter is not the Teacher used to build this target cache.\n"
            f"Expected SHA256 {expected_hash}\nGot      SHA256 {init_hash}"
        )

    max_train_steps = 1 if args.smoke_test else args.max_train_steps
    print("STUDENT TRAINING PLAN")
    print(f"Target cache    : {targets['path']}")
    print(f"Init adapter    : {init_adapter}")
    print(
        f"Distillation    : {target_metadata['source_steps']} -> "
        f"{target_metadata['student_steps']} DDIM steps"
    )
    print(f"Baked guidance  : {target_metadata['teacher_guidance']}")
    print(f"Target records  : {targets['noisy_latents'].shape[0]}")
    print(
        f"Target mode     : "
        f"{target_metadata.get('target_mode', 'forward_noised_clean_latent')}"
    )
    print(f"Optimizer steps : {max_train_steps}")
    print(f"Learning rate   : {args.learning_rate:g}")
    print(f"Fingerprint     : {training_cache['fingerprint']}")
    if args.check_only:
        print("CHECK-ONLY PASSED: target cache and adapter identity match.")
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to train the Student.")

    run_dir = build_run_dir(args, target_metadata)
    checkpoint_root = run_dir / "checkpoints"
    final_adapter = run_dir / "final_adapter"

    class DistillationDataset(Dataset):
        def __len__(self) -> int:
            return targets["noisy_latents"].shape[0]

        def __getitem__(self, index: int) -> dict:
            prompt_index = int(targets["sample_indices"][index])
            return {
                "noisy_latent": targets["noisy_latents"][index],
                "target_epsilon": targets["target_epsilons"][index],
                "timestep": targets["timesteps"][index],
                "prompt_embeds": training_cache["prompt_embeds"][prompt_index],
                "attention_mask": training_cache["attention_masks"][prompt_index],
            }

    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        DistillationDataset(),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=loader_generator,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("Loading Student from the Teacher adapter as trainable LoRA...")
    student, init_adapter = load_transformer_with_adapter(
        init_adapter, trainable=True, merge_for_inference=False
    )
    student.to("cuda")
    student.train()
    trainable_parameters = [p for p in student.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_parameters)
    if trainable_count == 0:
        raise RuntimeError("No trainable Student LoRA parameters found.")
    print(f"Trainable Student parameters: {trainable_count:,}")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    losses: list[float] = []
    started = time.perf_counter()
    progress = tqdm(
        total=max_train_steps,
        desc="Student optimizer updates",
        mininterval=15.0,
    )

    def save_checkpoint(step: int) -> Path:
        path = checkpoint_root / f"step_{step:06d}"
        student.save_pretrained(path, safe_serialization=True)
        metadata = {
            "optimizer_step": step,
            "distillation_loss_history": losses,
            "source_steps": target_metadata["source_steps"],
            "student_steps": target_metadata["student_steps"],
            "teacher_guidance_baked_in": target_metadata["teacher_guidance"],
            "inference_guidance": 1.0,
            "note": "Adapter-only checkpoint; optimizer state is not included.",
        }
        (path / "checkpoint_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return path

    while global_step < max_train_steps:
        for batch in loader:
            noisy = batch["noisy_latent"].to(
                "cuda", dtype=torch.float16, non_blocking=True
            )
            target = batch["target_epsilon"].to(
                "cuda", dtype=torch.float16, non_blocking=True
            )
            timestep = batch["timestep"].to("cuda", non_blocking=True)
            prompt = batch["prompt_embeds"].to(
                "cuda", dtype=torch.float16, non_blocking=True
            )
            mask = batch["attention_mask"].to("cuda", non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model_epsilon(
                    student,
                    noisy,
                    timestep,
                    prompt,
                    mask,
                    guidance_scale=1.0,
                )
                loss = F.mse_loss(
                    prediction.float(), target.float(), reduction="mean"
                )
                scaled_loss = loss / args.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss before optimizer step {global_step + 1}"
                )
            scaler.scale(scaled_loss).backward()
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, args.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                current_loss = loss.detach().item()
                losses.append(current_loss)
                progress.update(1)
                progress.set_postfix(loss=f"{current_loss:.6f}")
                if (
                    global_step % args.checkpointing_steps == 0
                    or global_step == max_train_steps
                ):
                    path = save_checkpoint(global_step)
                    print(f"\nSaved Student checkpoint: {path}")
            if global_step >= max_train_steps:
                break

    progress.close()
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    if len(losses) != global_step or not all(math.isfinite(x) for x in losses):
        raise RuntimeError("Student loss history validation failed.")
    student.save_pretrained(final_adapter, safe_serialization=True)

    metadata = {
        "status": "PASS",
        "transformer_model": TRANSFORMER_MODEL,
        "component_model": COMPONENT_MODEL,
        "target_cache": str(targets["path"]),
        "target_cache_metadata": target_metadata,
        "init_adapter": str(init_adapter),
        "init_adapter_sha256": init_hash,
        "source_steps": target_metadata["source_steps"],
        "student_steps": target_metadata["student_steps"],
        "teacher_guidance_baked_in": target_metadata["teacher_guidance"],
        "required_inference_guidance": 1.0,
        "scheduler": "DDIMScheduler",
        "timestep_spacing": "trailing",
        "optimizer": "torch.optim.AdamW",
        "optimizer_steps": global_step,
        "micro_steps": micro_step,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "trainable_parameters": trainable_count,
        "distillation_loss_history": losses,
        "elapsed_seconds": elapsed_seconds,
        "peak_vram_gb": peak_vram_gb,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }
    (run_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if not (final_adapter / "adapter_config.json").is_file() or not (
        final_adapter / "adapter_model.safetensors"
    ).is_file():
        raise FileNotFoundError("Final Student adapter was not saved correctly.")
    print("STUDENT TRAINING COMPLETE")
    print(f"Final adapter : {final_adapter}")
    print(f"Steps         : {global_step}")
    print(f"Elapsed       : {elapsed_seconds / 60:.1f} minutes")
    print(f"Peak VRAM     : {peak_vram_gb:.2f} GiB")
    print("Inference must use DDIM trailing and guidance_scale=1.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
