#!/usr/bin/env python3
"""Cache progressive targets from real PixArt Teacher sampling trajectories.

Unlike the original forward-noise cache, this script starts each trajectory
from random inference noise and follows the frozen Teacher's complete reverse
DDIM path. Every pair of Teacher steps becomes one target Student step.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from common import (
    COMPONENT_MODEL,
    TRANSFORMER_MODEL,
    derive_epsilon_for_endpoint,
    file_sha256,
    load_training_caches,
    load_transformer_with_adapter,
    make_ddim_scheduler,
    model_epsilon,
    progressive_transition_triples,
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
DEFAULT_TEACHER = (
    PROJECT_ROOT
    / "models"
    / "lora_training_512"
    / "style_teacher_r16_lr1e-5_bs1_steps10000_seed42"
    / "checkpoints"
    / "step_004000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-adapter", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--latent-cache", type=Path, default=DEFAULT_LATENTS)
    parser.add_argument("--text-cache", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-steps", type=int, default=20)
    parser.add_argument("--student-steps", type=int, default=10)
    parser.add_argument("--teacher-guidance", type=float, default=2.0)
    parser.add_argument("--trajectories-per-prompt", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit prompts for a smoke test; default uses all cached prompts.",
    )
    parser.add_argument(
        "--max-transitions",
        type=int,
        help="Limit paired transitions per trajectory for a smoke test.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate paths and schedules without loading the Teacher.",
    )
    return parser.parse_args()


def default_output(args: argparse.Namespace, prompt_count: int) -> Path:
    guidance = f"{args.teacher_guidance:g}".replace(".", "p")
    name = (
        f"teacher_trajectories_{args.source_steps}to{args.student_steps}_"
        f"g{guidance}_n{prompt_count}_r{args.trajectories_per_prompt}_"
        f"seed{args.seed}.pt"
    )
    return PROJECT_ROOT / "distillation" / "target_caches" / name


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() and not args.check_only:
        raise RuntimeError("CUDA is required to cache PixArt Teacher trajectories.")
    if args.teacher_guidance < 1.0:
        raise ValueError("--teacher-guidance must be at least 1.0")
    if args.trajectories_per_prompt < 1:
        raise ValueError("--trajectories-per-prompt must be positive")

    adapter = resolve_adapter(args.teacher_adapter)
    cache = load_training_caches(args.latent_cache, args.text_cache)
    transitions = progressive_transition_triples(
        args.source_steps, args.student_steps
    )
    prompt_count = len(cache["sample_ids"])
    if args.max_samples is not None:
        if not 1 <= args.max_samples <= prompt_count:
            raise ValueError(f"--max-samples must be 1..{prompt_count}")
        prompt_count = args.max_samples
    if args.max_transitions is not None:
        if not 1 <= args.max_transitions <= len(transitions):
            raise ValueError(f"--max-transitions must be 1..{len(transitions)}")
        transitions = transitions[: args.max_transitions]

    output = (
        args.output.expanduser().resolve()
        if args.output
        else default_output(args, prompt_count).resolve()
    )
    expected_targets = (
        prompt_count * args.trajectories_per_prompt * len(transitions)
    )
    print("TEACHER TRAJECTORY CACHE PLAN")
    print(f"Teacher adapter : {adapter}")
    print(f"Teacher schedule: {args.source_steps} DDIM trailing steps")
    print(f"Student schedule: {args.student_steps} DDIM trailing steps")
    print(f"Teacher guidance: {args.teacher_guidance:g}")
    print(f"Prompts         : {prompt_count}")
    print(f"Trajectories    : {args.trajectories_per_prompt} per prompt")
    print(f"Transitions     : {transitions}")
    print(f"Target records  : {expected_targets}")
    print(f"Output          : {output}")
    print(f"Fingerprint     : {cache['fingerprint']}")
    if args.check_only:
        print("CHECK-ONLY PASSED: no model loaded and no targets generated.")
        return 0
    if output.exists():
        raise FileExistsError(
            f"Target cache already exists: {output}\nChoose a new --output path."
        )

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    source_scheduler = make_ddim_scheduler(args.source_steps, device=device)
    student_scheduler = make_ddim_scheduler(args.student_steps, device=device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("Loading frozen Teacher transformer and merging its LoRA...")
    teacher, adapter = load_transformer_with_adapter(
        adapter, trainable=False, merge_for_inference=True
    )
    teacher.requires_grad_(False)
    teacher.to(device).eval()

    empty_embeds = cache["empty_prompt_embeds"].to(
        device, dtype=torch.float16
    )
    empty_mask = cache["empty_prompt_attention_mask"].to(device)
    noisy_latents: list[torch.Tensor] = []
    target_epsilons: list[torch.Tensor] = []
    sample_indices: list[int] = []
    trajectory_indices: list[int] = []
    trajectory_seeds: list[int] = []
    timesteps: list[int] = []
    destination_timesteps: list[int] = []
    max_endpoint_error = 0.0
    progress = tqdm(
        total=expected_targets,
        desc="Teacher trajectory targets",
        mininterval=15.0,
    )
    started = time.perf_counter()

    with torch.inference_mode():
        for trajectory_index in range(args.trajectories_per_prompt):
            for sample_index in range(prompt_count):
                prompt = cache["prompt_embeds"][
                    sample_index : sample_index + 1
                ].to(device, dtype=torch.float16, non_blocking=True)
                mask = cache["attention_masks"][
                    sample_index : sample_index + 1
                ].to(device, non_blocking=True)
                trajectory_seed = (
                    args.seed + trajectory_index * prompt_count + sample_index
                )
                generator = torch.Generator(device="cpu").manual_seed(
                    trajectory_seed
                )
                sample_t = torch.randn(
                    (1, 4, 64, 64), generator=generator, dtype=torch.float32
                ).to(device=device, dtype=torch.float16)
                sample_t = sample_t * source_scheduler.init_noise_sigma

                for timestep, middle_timestep, destination_timestep in transitions:
                    transition_start = sample_t
                    teacher_epsilon_1 = model_epsilon(
                        teacher,
                        transition_start,
                        timestep,
                        prompt,
                        mask,
                        guidance_scale=args.teacher_guidance,
                        empty_prompt_embeds=empty_embeds,
                        empty_attention_mask=empty_mask,
                    )
                    sample_middle = source_scheduler.step(
                        teacher_epsilon_1,
                        timestep,
                        transition_start,
                        eta=0.0,
                        return_dict=True,
                    ).prev_sample
                    teacher_epsilon_2 = model_epsilon(
                        teacher,
                        sample_middle,
                        middle_timestep,
                        prompt,
                        mask,
                        guidance_scale=args.teacher_guidance,
                        empty_prompt_embeds=empty_embeds,
                        empty_attention_mask=empty_mask,
                    )
                    teacher_endpoint = source_scheduler.step(
                        teacher_epsilon_2,
                        middle_timestep,
                        sample_middle,
                        eta=0.0,
                        return_dict=True,
                    ).prev_sample
                    target_epsilon = derive_epsilon_for_endpoint(
                        student_scheduler,
                        transition_start,
                        teacher_endpoint,
                        timestep,
                        destination_timestep,
                    )
                    reproduced_endpoint = student_scheduler.step(
                        target_epsilon,
                        timestep,
                        transition_start,
                        eta=0.0,
                        return_dict=True,
                    ).prev_sample
                    endpoint_error = (
                        reproduced_endpoint.float() - teacher_endpoint.float()
                    ).abs().max().item()
                    max_endpoint_error = max(max_endpoint_error, endpoint_error)
                    if not torch.isfinite(target_epsilon).all():
                        raise FloatingPointError(
                            "Non-finite target at "
                            f"sample={sample_index}, trajectory={trajectory_index}, "
                            f"t={timestep}"
                        )

                    noisy_latents.append(transition_start.squeeze(0).cpu())
                    target_epsilons.append(target_epsilon.squeeze(0).cpu())
                    sample_indices.append(sample_index)
                    trajectory_indices.append(trajectory_index)
                    trajectory_seeds.append(trajectory_seed)
                    timesteps.append(timestep)
                    destination_timesteps.append(destination_timestep)
                    sample_t = teacher_endpoint
                    progress.update(1)

    progress.close()
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    payload = {
        "format": "pixart_progressive_distillation_targets_v2_teacher_trajectory",
        "noisy_latents": torch.stack(noisy_latents).contiguous(),
        "target_epsilons": torch.stack(target_epsilons).contiguous(),
        "sample_indices": torch.tensor(sample_indices, dtype=torch.int64),
        "trajectory_indices": torch.tensor(
            trajectory_indices, dtype=torch.int64
        ),
        "trajectory_seeds": torch.tensor(trajectory_seeds, dtype=torch.int64),
        "timesteps": torch.tensor(timesteps, dtype=torch.int64),
        "destination_timesteps": torch.tensor(
            destination_timesteps, dtype=torch.int64
        ),
        "metadata": {
            "target_mode": "teacher_reverse_sampling_trajectory",
            "transformer_model": TRANSFORMER_MODEL,
            "component_model": COMPONENT_MODEL,
            "teacher_adapter": str(adapter),
            "teacher_adapter_sha256": file_sha256(
                adapter / "adapter_model.safetensors"
            ),
            "source_steps": args.source_steps,
            "student_steps": args.student_steps,
            "teacher_guidance": args.teacher_guidance,
            "scheduler": "DDIMScheduler",
            "timestep_spacing": "trailing",
            "eta": 0.0,
            "prediction_type": "epsilon",
            "seed": args.seed,
            "num_source_samples": prompt_count,
            "trajectories_per_prompt": args.trajectories_per_prompt,
            "transition_triples": transitions,
            "num_targets": expected_targets,
            "manifest_fingerprint": cache["fingerprint"],
            "latent_cache": str(cache["latent_path"]),
            "text_cache": str(cache["text_path"]),
            "max_endpoint_reproduction_error": max_endpoint_error,
            "elapsed_seconds": elapsed_seconds,
            "peak_vram_gb": peak_vram_gb,
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
    }
    if payload["noisy_latents"].shape != payload["target_epsilons"].shape:
        raise RuntimeError("Target cache tensor shapes do not match.")
    if payload["noisy_latents"].shape[0] != expected_targets:
        raise RuntimeError("Target cache record count does not match the plan.")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(payload["metadata"], indent=2), encoding="utf-8"
    )
    print("TEACHER TRAJECTORY CACHE COMPLETE")
    print(f"Targets      : {output}")
    print(f"Metadata     : {metadata_path}")
    print(f"Records      : {expected_targets}")
    print(f"Endpoint err : {max_endpoint_error:.6g}")
    print(f"Elapsed      : {elapsed_seconds / 60:.1f} minutes")
    print(f"Peak VRAM    : {peak_vram_gb:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
