#!/usr/bin/env python3
"""Cache deterministic 20-step PixArt teacher trajectories."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import save_file

from scripts.distillation.common import (
    COMPONENT_MODEL,
    LATENT_SHAPE,
    TEACHER_TIMESTEPS,
    TRANSFORMER_MODEL,
    deterministic_trajectory_seed,
    load_distill_prompt_cache,
    repository_root,
    resolve_adapter_dir,
    sha256_file,
    split_epsilon_prediction,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description="Generate sharded 20-step teacher trajectory states."
    )
    parser.add_argument("--teacher-manifest", type=Path, required=True)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicas-per-prompt", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument(
        "--limit-trajectories",
        type=int,
        default=None,
        help="Limit cache size for smoke testing.",
    )
    parser.add_argument("--transformer-model", default=TRANSFORMER_MODEL)
    parser.add_argument("--component-model", default=COMPONENT_MODEL)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_teacher_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "PASS":
        raise ValueError("Teacher manifest must be a PASS JSON object.")
    required = {
        "teacher_id",
        "adapter_dir",
        "adapter_sha256",
        "teacher_guidance_scale",
        "transformer_model",
        "component_model",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Teacher manifest is missing keys: {missing}")
    adapter = resolve_adapter_dir(manifest["adapter_dir"])
    if sha256_file(adapter / "adapter_model.safetensors") != manifest[
        "adapter_sha256"
    ]:
        raise ValueError("Teacher adapter SHA256 no longer matches manifest.")
    return manifest


def _trajectory_plan(
    prompt_ids: Sequence[str], replicas: int
) -> list[dict[str, int | str]]:
    return [
        {
            "prompt_index": prompt_index,
            "prompt_id": prompt_id,
            "replica": replica,
            "seed": deterministic_trajectory_seed(prompt_id, replica),
        }
        for prompt_index, prompt_id in enumerate(prompt_ids)
        for replica in range(replicas)
    ]


def _validate_existing_manifest(
    existing: dict[str, Any],
    teacher: dict[str, Any],
    prompt_fingerprint: str,
    replicas: int,
) -> None:
    expected = {
        "teacher_id": teacher["teacher_id"],
        "teacher_adapter_sha256": teacher["adapter_sha256"],
        "teacher_guidance_scale": teacher["teacher_guidance_scale"],
        "prompt_bank_fingerprint": prompt_fingerprint,
        "replicas_per_prompt": replicas,
        "teacher_timesteps": list(TEACHER_TIMESTEPS),
    }
    for key, value in expected.items():
        if existing.get(key) != value:
            raise ValueError(
                f"Existing trajectory cache {key!r} mismatch: "
                f"expected {value!r}, got {existing.get(key)!r}."
            )


def _generate_one(
    *,
    transformer: torch.nn.Module,
    scheduler: Any,
    prompt_embed: torch.Tensor,
    prompt_mask: torch.Tensor,
    empty_embed: torch.Tensor,
    empty_mask: torch.Tensor,
    seed: int,
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    scheduler.set_timesteps(20, device=device)
    actual_timesteps = tuple(int(value) for value in scheduler.timesteps.tolist())
    if actual_timesteps != TEACHER_TIMESTEPS:
        raise RuntimeError(
            "Teacher scheduler timestep mismatch: "
            f"expected {TEACHER_TIMESTEPS}, got {actual_timesteps}."
        )
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, *LATENT_SHAPE),
        generator=generator,
        device=device,
        dtype=torch.float16,
    ) * scheduler.init_noise_sigma
    do_cfg = guidance_scale > 1.0
    if do_cfg:
        embeddings = torch.cat([empty_embed, prompt_embed])
        masks = torch.cat([empty_mask, prompt_mask])
    else:
        embeddings = prompt_embed
        masks = prompt_mask
    states = [latents.detach().to("cpu", dtype=torch.float16)]
    with torch.inference_mode():
        for timestep in scheduler.timesteps:
            model_input = torch.cat([latents, latents]) if do_cfg else latents
            model_input = scheduler.scale_model_input(model_input, timestep)
            timestep_batch = timestep.reshape(1).expand(model_input.shape[0])
            output = transformer(
                model_input,
                encoder_hidden_states=embeddings,
                encoder_attention_mask=masks,
                timestep=timestep_batch,
                added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                return_dict=False,
            )[0]
            if do_cfg:
                output_uncond, output_cond = output.chunk(2)
                output = output_uncond + guidance_scale * (
                    output_cond - output_uncond
                )
            epsilon = split_epsilon_prediction(output)
            latents = scheduler.step(
                epsilon,
                timestep,
                latents,
                return_dict=False,
            )[0]
            if not bool(torch.isfinite(latents).all()):
                raise FloatingPointError(
                    f"Non-finite teacher latent after timestep {int(timestep)}."
                )
            states.append(latents.detach().to("cpu", dtype=torch.float16))
    return torch.cat(states, dim=0)


def cache_trajectories(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to cache trajectories.")
    if args.replicas_per_prompt <= 0 or args.shard_size <= 0:
        raise ValueError("Replica and shard sizes must be positive.")
    teacher = _load_teacher_manifest(args.teacher_manifest.resolve())
    if teacher["transformer_model"] != args.transformer_model:
        raise ValueError("Teacher transformer model does not match CLI model.")
    if teacher["component_model"] != args.component_model:
        raise ValueError("Teacher component model does not match CLI model.")
    prompts = load_distill_prompt_cache(args.prompt_cache)
    plan = _trajectory_plan(prompts.prompt_ids, args.replicas_per_prompt)
    full_count = len(plan)
    if args.limit_trajectories is not None:
        if args.limit_trajectories <= 0:
            raise ValueError("--limit-trajectories must be positive.")
        plan = plan[: args.limit_trajectories]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "cache_manifest.json"
    existing: dict[str, Any] | None = None
    if manifest_path.is_file() and not args.overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_existing_manifest(
            existing,
            teacher,
            prompts.prompt_bank_fingerprint,
            args.replicas_per_prompt,
        )
        completed = int(existing.get("trajectory_count", 0))
        shards = list(existing.get("shards", []))
        for shard in shards:
            shard_path = output_dir / shard["file"]
            if not shard_path.is_file() or sha256_file(shard_path) != shard[
                "sha256"
            ]:
                raise ValueError(f"Existing cache shard failed SHA: {shard_path}")
        if completed >= len(plan):
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return existing
    else:
        completed = 0
        shards = []

    from diffusers import DPMSolverMultistepScheduler, PixArtTransformer2DModel
    from peft import PeftModel

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    base = PixArtTransformer2DModel.from_pretrained(
        args.transformer_model,
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    )
    transformer = PeftModel.from_pretrained(
        base,
        resolve_adapter_dir(teacher["adapter_dir"]),
        is_trainable=False,
    ).to(device).eval()
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.component_model,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )
    guidance_scale = float(teacher["teacher_guidance_scale"])
    empty_embed = prompts.empty_prompt_embeds.to(device)
    empty_mask = prompts.empty_prompt_attention_mask.to(device)
    started = time.perf_counter()

    while completed < len(plan):
        shard_plan = plan[completed : completed + args.shard_size]
        shard_states: list[torch.Tensor] = []
        prompt_indices: list[int] = []
        seeds: list[int] = []
        replicas: list[int] = []
        for record in shard_plan:
            prompt_index = int(record["prompt_index"])
            states = _generate_one(
                transformer=transformer,
                scheduler=scheduler,
                prompt_embed=prompts.prompt_embeds[
                    prompt_index : prompt_index + 1
                ].to(device),
                prompt_mask=prompts.attention_masks[
                    prompt_index : prompt_index + 1
                ].to(device),
                empty_embed=empty_embed,
                empty_mask=empty_mask,
                seed=int(record["seed"]),
                guidance_scale=guidance_scale,
                device=device,
            )
            shard_states.append(states.unsqueeze(0))
            prompt_indices.append(prompt_index)
            seeds.append(int(record["seed"]))
            replicas.append(int(record["replica"]))
            print(
                f"trajectory={completed + len(shard_states)}/{len(plan)} "
                f"prompt={record['prompt_id']} seed={record['seed']}"
            )
        shard_number = len(shards)
        shard_name = f"trajectories-{shard_number:05d}.safetensors"
        shard_path = output_dir / shard_name
        tensors = {
            "states": torch.cat(shard_states).contiguous(),
            "prompt_indices": torch.tensor(prompt_indices, dtype=torch.int64),
            "seeds": torch.tensor(seeds, dtype=torch.int64),
            "replicas": torch.tensor(replicas, dtype=torch.int64),
        }
        if tensors["states"].shape[1:] != (21, *LATENT_SHAPE):
            raise RuntimeError(f"Unexpected trajectory shape: {tensors['states'].shape}")
        if not bool(torch.isfinite(tensors["states"]).all()):
            raise FloatingPointError("Trajectory shard contains non-finite values.")
        save_file(
            tensors,
            shard_path,
            metadata={
                "format_version": "1",
                "teacher_id": str(teacher["teacher_id"]),
                "prompt_bank_fingerprint": prompts.prompt_bank_fingerprint,
            },
        )
        shard_record = {
            "file": shard_name,
            "sha256": sha256_file(shard_path),
            "start_index": completed,
            "count": len(shard_plan),
            "states_shape": list(tensors["states"].shape),
            "states_dtype": str(tensors["states"].dtype),
        }
        shards.append(shard_record)
        completed += len(shard_plan)
        status = "PASS" if completed == full_count else "PARTIAL"
        manifest: dict[str, Any] = {
            "format_version": 1,
            "status": status,
            "teacher_id": teacher["teacher_id"],
            "teacher_manifest": str(args.teacher_manifest.resolve()),
            "teacher_adapter_sha256": teacher["adapter_sha256"],
            "teacher_guidance_scale": guidance_scale,
            "transformer_model": args.transformer_model,
            "component_model": args.component_model,
            "scheduler_class": "DPMSolverMultistepScheduler",
            "teacher_timesteps": list(TEACHER_TIMESTEPS),
            "states_per_trajectory": 21,
            "latent_shape": list(LATENT_SHAPE),
            "latent_dtype": "torch.float16",
            "prompt_cache": str(args.prompt_cache.resolve()),
            "prompt_bank_fingerprint": prompts.prompt_bank_fingerprint,
            "prompt_count": len(prompts.prompt_ids),
            "replicas_per_prompt": args.replicas_per_prompt,
            "planned_trajectory_count": full_count,
            "trajectory_count": completed,
            "limited_run": len(plan) != full_count,
            "transformer_forward_calls": completed * 20,
            "shards": shards,
            "elapsed_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
        }
        write_json(manifest_path, manifest)

    del transformer, base
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_trajectories(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
