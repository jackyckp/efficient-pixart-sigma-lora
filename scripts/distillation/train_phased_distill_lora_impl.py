#!/usr/bin/env python3
"""Train a joint PixArt style-and-speed LoRA from trajectory shards."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from scripts.distillation.common import (
    COMPONENT_MODEL,
    LATENT_SHAPE,
    TRANSFORMER_MODEL,
    deterministic_jump,
    load_distill_prompt_cache,
    phase_pairs,
    pseudo_huber_loss,
    repository_root,
    resolve_adapter_dir,
    sha256_file,
    split_epsilon_prediction,
    state_timestep,
    write_json,
)
from scripts.training.train_local_latent_lora import load_latent_bundle


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description="Train a 4-step or 2-step joint PixArt LoRA student."
    )
    parser.add_argument("--trajectory-cache", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, choices=(2, 4), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-cache",
        type=Path,
        default=(
            root
            / "data"
            / "features"
            / "distill_t5_plant627_len300_fp16_v1.pt"
        ),
    )
    parser.add_argument(
        "--latent-bundle",
        type=Path,
        default=root / "data" / "archives" / "clean_latents_512.zip",
    )
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    parser.add_argument("--anchor-probability", type=float, default=0.2)
    parser.add_argument("--on-policy-probability", type=float, default=0.5)
    parser.add_argument("--huber-c", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--transformer-model", default=TRANSFORMER_MODEL)
    parser.add_argument("--component-model", default=COMPONENT_MODEL)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-partial-cache",
        action="store_true",
        help="Permit a limited trajectory cache for smoke tests.",
    )
    return parser


def apply_stage_defaults(args: argparse.Namespace) -> None:
    if args.target_steps == 4:
        args.max_train_steps = args.max_train_steps or 2_000
        args.learning_rate = args.learning_rate or 5e-6
        args.checkpoint_every_steps = args.checkpoint_every_steps or 500
    else:
        args.max_train_steps = args.max_train_steps or 10_000
        args.learning_rate = args.learning_rate or 2e-6
        args.checkpoint_every_steps = args.checkpoint_every_steps or 1_000


def validate_args(args: argparse.Namespace) -> None:
    apply_stage_defaults(args)
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"Expected Python 3.11.x, got {platform.python_version()}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for distillation training.")
    for name in ("max_train_steps", "checkpoint_every_steps", "log_every_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("learning_rate", "huber_c", "max_grad_norm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("anchor_probability", "on_policy_probability"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")


def load_trajectory_cache(
    cache_dir: Path,
    *,
    allow_partial: bool,
    expected_prompt_fingerprint: str,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    manifest_path = cache_dir / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_statuses = {"PASS", "PARTIAL"} if allow_partial else {"PASS"}
    if manifest.get("status") not in allowed_statuses:
        raise ValueError(
            f"Trajectory cache status must be {sorted(allowed_statuses)}, "
            f"got {manifest.get('status')!r}."
        )
    if manifest.get("prompt_bank_fingerprint") != expected_prompt_fingerprint:
        raise ValueError("Trajectory/prompt cache fingerprint mismatch.")
    if manifest.get("states_per_trajectory") != 21:
        raise ValueError("Trajectory cache must contain 21 states per record.")
    all_states: list[torch.Tensor] = []
    all_prompt_indices: list[torch.Tensor] = []
    all_seeds: list[torch.Tensor] = []
    for shard in manifest.get("shards", []):
        path = cache_dir / shard["file"]
        if not path.is_file() or sha256_file(path) != shard["sha256"]:
            raise ValueError(f"Trajectory shard SHA mismatch: {path}")
        tensors = load_file(path, device="cpu")
        states = tensors["states"]
        if states.ndim != 6 or states.shape[1:] != (21, *LATENT_SHAPE):
            raise ValueError(f"Invalid trajectory states shape in {path}.")
        if states.dtype != torch.float16 or not bool(torch.isfinite(states).all()):
            raise ValueError(f"Invalid trajectory states values in {path}.")
        all_states.append(states)
        all_prompt_indices.append(tensors["prompt_indices"].to(torch.int64))
        all_seeds.append(tensors["seeds"].to(torch.int64))
    if not all_states:
        raise ValueError("Trajectory cache has no shards.")
    states = torch.cat(all_states).contiguous()
    prompt_indices = torch.cat(all_prompt_indices).contiguous()
    seeds = torch.cat(all_seeds).contiguous()
    if len(states) != manifest.get("trajectory_count"):
        raise ValueError("Trajectory manifest count does not match shards.")
    return manifest, states, prompt_indices, seeds


def _forward_epsilon(
    transformer: torch.nn.Module,
    latent: torch.Tensor,
    prompt_embed: torch.Tensor,
    prompt_mask: torch.Tensor,
    timestep: int | torch.Tensor,
) -> torch.Tensor:
    if isinstance(timestep, int):
        timesteps = torch.full(
            (latent.shape[0],),
            timestep,
            device=latent.device,
            dtype=torch.long,
        )
    else:
        timesteps = timestep.to(latent.device, dtype=torch.long)
    output = transformer(
        latent,
        encoder_hidden_states=prompt_embed,
        encoder_attention_mask=prompt_mask,
        timestep=timesteps,
        added_cond_kwargs={"resolution": None, "aspect_ratio": None},
        return_dict=False,
    )[0]
    return split_epsilon_prediction(output)


def _save_checkpoint(
    *,
    accelerator: Any,
    transformer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    global_step: int,
    loss_history: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    best: bool,
) -> Path:
    checkpoint = output_dir / "checkpoints" / f"step_{global_step:06d}"
    adapter_dir = checkpoint / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(transformer)
    unwrapped.save_pretrained(adapter_dir, safe_serialization=True)
    state = {
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
        "loss_history": list(loss_history),
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all(),
    }
    torch.save(state, checkpoint / "training_state.pt")
    write_json(
        checkpoint / "checkpoint_metadata.json",
        {**metadata, "status": "CHECKPOINT", "optimizer_step": global_step},
    )
    if best:
        best_dir = output_dir / "best_adapter"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(adapter_dir, best_dir)
        write_json(
            output_dir / "best_checkpoint.json",
            {
                "optimizer_step": global_step,
                "checkpoint": str(checkpoint),
                "selection_metric": "mean checkpoint interval training loss",
                "note": "Run external CLIP/CMMD evaluation before final selection.",
            },
        )
    return checkpoint


def train(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import DDPMScheduler, PixArtTransformer2DModel
    from peft import PeftModel

    set_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    prompts = load_distill_prompt_cache(args.prompt_cache)
    cache_dir = args.trajectory_cache.resolve()
    cache_manifest, states, trajectory_prompt_indices, trajectory_seeds = (
        load_trajectory_cache(
            cache_dir,
            allow_partial=args.allow_partial_cache,
            expected_prompt_fingerprint=prompts.prompt_bank_fingerprint,
        )
    )
    if cache_manifest["transformer_model"] != args.transformer_model:
        raise ValueError("Trajectory transformer model mismatch.")
    if cache_manifest["component_model"] != args.component_model:
        raise ValueError("Trajectory component model mismatch.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_state = None
    init_adapter = resolve_adapter_dir(args.init_adapter)
    if args.resume_from is not None:
        resume_dir = args.resume_from.resolve()
        init_adapter = resolve_adapter_dir(resume_dir)
        resume_state = torch.load(
            resume_dir / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )

    base = PixArtTransformer2DModel.from_pretrained(
        args.transformer_model,
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    )
    base.requires_grad_(False)
    base.enable_gradient_checkpointing()
    transformer = PeftModel.from_pretrained(
        base,
        init_adapter,
        is_trainable=True,
    )
    if transformer.peft_config["default"].r != 16:
        raise RuntimeError("Joint student must remain rank 16.")
    for parameter in transformer.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    trainable = [
        parameter for parameter in transformer.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters were loaded.")

    scheduler = DDPMScheduler.from_pretrained(
        args.component_model,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )
    if scheduler.config.prediction_type != "epsilon":
        raise RuntimeError("Phased trainer currently requires epsilon prediction.")
    alphas = scheduler.alphas_cumprod.to(torch.float32)
    latent_bundle = load_latent_bundle(args.latent_bundle)
    latent_index = {
        sample_id: index for index, sample_id in enumerate(latent_bundle.sample_ids)
    }
    original_prompt_indices = [
        index
        for index, prompt_id in enumerate(prompts.prompt_ids)
        if prompt_id.endswith("::original")
    ]
    if len(original_prompt_indices) != 209:
        raise ValueError("Expected 209 original prompt records for anchor loss.")
    for index in original_prompt_indices:
        if prompts.source_sample_ids[index] not in latent_index:
            raise ValueError("Anchor prompt source is absent from latent bundle.")

    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=1)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    transformer, optimizer = accelerator.prepare(transformer, optimizer)
    device = accelerator.device

    global_step = 0
    loss_history: list[dict[str, Any]] = []
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        global_step = int(resume_state["global_step"])
        loss_history = list(resume_state["loss_history"])
        random.setstate(resume_state["python_random_state"])
        torch.set_rng_state(resume_state["torch_random_state"])
        torch.cuda.set_rng_state_all(resume_state["cuda_random_state"])
    if global_step >= args.max_train_steps:
        raise ValueError("Resume checkpoint already reached max training steps.")

    metadata_base = {
        "format_version": 1,
        "run_role": "phased_joint_lora_student",
        "teacher_id": cache_manifest["teacher_id"],
        "teacher_adapter_sha256": cache_manifest["teacher_adapter_sha256"],
        "trajectory_cache": str(cache_dir),
        "trajectory_count": len(states),
        "prompt_cache": str(args.prompt_cache.resolve()),
        "prompt_bank_fingerprint": prompts.prompt_bank_fingerprint,
        "init_adapter": str(init_adapter),
        "init_adapter_sha256": sha256_file(
            init_adapter / "adapter_model.safetensors"
        ),
        "target_inference_steps": args.target_steps,
        "student_guidance_scale": 1.0,
        "classifier_free_guidance_branch": False,
        "phase_index_pairs": [list(pair) for pair in phase_pairs(args.target_steps)],
        "rank": 16,
        "lora_alpha": 16,
        "learning_rate": args.learning_rate,
        "max_train_steps": args.max_train_steps,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "anchor_probability": args.anchor_probability,
        "on_policy_probability": (
            args.on_policy_probability if args.target_steps == 2 else 0.0
        ),
        "huber_c": args.huber_c,
        "seed": args.seed,
        "transformer_model": args.transformer_model,
        "component_model": args.component_model,
    }
    write_json(output_dir / "run_config.json", metadata_base)

    checkpoint_losses: list[float] = []
    best_interval_loss = math.inf
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    transformer.train()
    phase_options = phase_pairs(args.target_steps)
    while global_step < args.max_train_steps:
        use_anchor = random.random() < args.anchor_probability
        optimizer.zero_grad(set_to_none=True)
        if use_anchor:
            prompt_index = random.choice(original_prompt_indices)
            source_id = prompts.source_sample_ids[prompt_index]
            clean = latent_bundle.latents[
                latent_index[source_id] : latent_index[source_id] + 1
            ].to(device, dtype=torch.float16)
            noise = torch.randn_like(clean)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (1,),
                device=device,
                dtype=torch.long,
            )
            noisy = scheduler.add_noise(clean, noise, timesteps)
            prompt_embed = prompts.prompt_embeds[
                prompt_index : prompt_index + 1
            ].to(device)
            prompt_mask = prompts.attention_masks[
                prompt_index : prompt_index + 1
            ].to(device)
            with accelerator.autocast():
                epsilon = _forward_epsilon(
                    transformer, noisy, prompt_embed, prompt_mask, timesteps
                )
                loss = F.mse_loss(epsilon.float(), noise.float())
            loss_kind = "anchor"
            phase_number = None
            used_on_policy = False
        else:
            trajectory_index = random.randrange(len(states))
            phase_number = random.randrange(len(phase_options))
            start_index, target_index = phase_options[phase_number]
            prompt_index = int(trajectory_prompt_indices[trajectory_index])
            prompt_embed = prompts.prompt_embeds[
                prompt_index : prompt_index + 1
            ].to(device)
            prompt_mask = prompts.attention_masks[
                prompt_index : prompt_index + 1
            ].to(device)
            start = states[
                trajectory_index, start_index : start_index + 1
            ].to(device)
            target = states[
                trajectory_index, target_index : target_index + 1
            ].to(device)
            start_t = state_timestep(start_index)
            target_t = state_timestep(target_index)
            used_on_policy = (
                args.target_steps == 2
                and phase_number == 1
                and random.random() < args.on_policy_probability
            )
            if used_on_policy:
                first_start_index, first_target_index = phase_options[0]
                initial = states[
                    trajectory_index,
                    first_start_index : first_start_index + 1,
                ].to(device)
                with torch.no_grad(), accelerator.autocast():
                    first_epsilon = _forward_epsilon(
                        transformer,
                        initial,
                        prompt_embed,
                        prompt_mask,
                        state_timestep(first_start_index),
                    )
                    start = deterministic_jump(
                        initial,
                        first_epsilon,
                        state_timestep(first_start_index),
                        state_timestep(first_target_index),
                        alphas,
                    ).to(dtype=torch.float16).detach()
            with accelerator.autocast():
                epsilon = _forward_epsilon(
                    transformer, start, prompt_embed, prompt_mask, start_t
                )
                predicted_target = deterministic_jump(
                    start,
                    epsilon,
                    start_t,
                    target_t,
                    alphas,
                )
                loss = pseudo_huber_loss(
                    predicted_target,
                    target,
                    c=args.huber_c,
                )
            loss_kind = "trajectory"

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite loss at optimizer step {global_step}."
            )
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
        optimizer.step()
        global_step += 1
        loss_value = float(loss.detach().item())
        checkpoint_losses.append(loss_value)
        loss_history.append(
            {
                "step": global_step,
                "loss": loss_value,
                "kind": loss_kind,
                "phase": phase_number,
                "on_policy": used_on_policy,
            }
        )
        if (
            global_step == 1
            or global_step % args.log_every_steps == 0
            or global_step == args.max_train_steps
        ):
            print(
                f"optimizer_step={global_step}/{args.max_train_steps} "
                f"loss={loss_value:.6f} kind={loss_kind} "
                f"phase={phase_number} on_policy={used_on_policy}"
            )
        should_checkpoint = (
            global_step % args.checkpoint_every_steps == 0
            or global_step == args.max_train_steps
        )
        if should_checkpoint and accelerator.is_main_process:
            interval_loss = sum(checkpoint_losses) / len(checkpoint_losses)
            is_best = interval_loss < best_interval_loss
            if is_best:
                best_interval_loss = interval_loss
            checkpoint = _save_checkpoint(
                accelerator=accelerator,
                transformer=transformer,
                optimizer=optimizer,
                output_dir=output_dir,
                global_step=global_step,
                loss_history=loss_history,
                metadata={
                    **metadata_base,
                    "interval_mean_loss": interval_loss,
                    "best_interval_mean_loss": best_interval_loss,
                },
                best=is_best,
            )
            print(f"saved_checkpoint={checkpoint}")
            checkpoint_losses.clear()

    accelerator.wait_for_everyone()
    final_adapter = output_dir / "lora_adapter"
    if accelerator.is_main_process:
        accelerator.unwrap_model(transformer).save_pretrained(
            final_adapter,
            safe_serialization=True,
        )
    accelerator.wait_for_everyone()
    elapsed = time.perf_counter() - started
    result = {
        **metadata_base,
        "status": "PASS",
        "optimizer_steps": global_step,
        "finite_loss_count": len(loss_history),
        "final_loss": loss_history[-1]["loss"],
        "best_interval_mean_loss": best_interval_loss,
        "train_seconds": elapsed,
        "seconds_per_optimizer_step": elapsed / global_step,
        "peak_allocated_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "final_adapter": str(final_adapter),
        "final_adapter_sha256": sha256_file(
            final_adapter / "adapter_model.safetensors"
        ),
        "fresh_reload_required": True,
        "evaluation_status": "PENDING",
    }
    write_json(output_dir / "run_metadata.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    del optimizer, transformer, base, states
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
