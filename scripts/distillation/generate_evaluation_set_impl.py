#!/usr/bin/env python3
"""Generate matched teacher/student images for distillation evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from scripts.distillation.common import (
    COMPONENT_MODEL,
    LATENT_SHAPE,
    TRANSFORMER_MODEL,
    deterministic_jump,
    phase_pairs,
    resolve_adapter_dir,
    split_epsilon_prediction,
    state_timestep,
    write_json,
)
from scripts.distillation.evaluation_prompt_cache import (
    DEFAULT_CACHE,
    load_evaluation_prompt_cache,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate matched 20-step and distilled evaluation sets."
    )
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--student-adapter", type=Path, required=True)
    parser.add_argument("--student-steps", type=int, choices=(2, 4), required=True)
    parser.add_argument("--evaluation-prompts", type=Path, required=True)
    parser.add_argument(
        "--evaluation-prompt-cache", type=Path, default=DEFAULT_CACHE
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-limit", type=int, default=30)
    parser.add_argument("--seeds-per-prompt", type=int, default=4)
    parser.add_argument("--transformer-model", default=TRANSFORMER_MODEL)
    parser.add_argument("--component-model", default=COMPONENT_MODEL)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload = json.loads(args.evaluation_prompts.read_text(encoding="utf-8"))
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Evaluation prompt manifest contains no prompts.")
    selected = prompts[: args.prompt_limit]
    records: list[dict[str, Any]] = []
    for item in selected:
        for seed in item["seeds"][: args.seeds_per_prompt]:
            records.append(
                {
                    "prompt_id": item["prompt_id"],
                    "prompt": item["prompt"],
                    "seed": int(seed),
                    "filename": f"{item['prompt_id']}_seed{int(seed)}.png",
                }
            )
    return records


def _encode_unique_prompts(
    records: Sequence[dict[str, Any]], args: argparse.Namespace
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return load_evaluation_prompt_cache(
        args.evaluation_prompt_cache,
        records,
        component_model=args.component_model,
    )


def _generate_teacher_final(
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
    """Run the exact 20-call teacher schedule and retain only its final state."""
    from scripts.distillation.common import TEACHER_TIMESTEPS

    scheduler.set_timesteps(20, device=device)
    if tuple(int(value) for value in scheduler.timesteps.tolist()) != TEACHER_TIMESTEPS:
        raise RuntimeError("Teacher evaluation timestep schedule changed.")
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(
        (1, *LATENT_SHAPE), generator=generator, device=device, dtype=torch.float16
    ) * scheduler.init_noise_sigma
    use_cfg = guidance_scale > 1.0
    embeds = torch.cat([empty_embed, prompt_embed]) if use_cfg else prompt_embed
    masks = torch.cat([empty_mask, prompt_mask]) if use_cfg else prompt_mask
    with torch.inference_mode():
        for timestep in scheduler.timesteps:
            model_input = torch.cat([latent, latent]) if use_cfg else latent
            model_input = scheduler.scale_model_input(model_input, timestep)
            output = transformer(
                model_input,
                encoder_hidden_states=embeds,
                encoder_attention_mask=masks,
                timestep=timestep.reshape(1).expand(model_input.shape[0]),
                added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                return_dict=False,
            )[0]
            if use_cfg:
                unconditional, conditional = output.chunk(2)
                output = unconditional + guidance_scale * (
                    conditional - unconditional
                )
            latent = scheduler.step(
                split_epsilon_prediction(output), timestep, latent, return_dict=False
            )[0]
    if not bool(torch.isfinite(latent).all()):
        raise FloatingPointError("Teacher evaluation latent is not finite.")
    return latent.to("cpu", dtype=torch.float16)


def _student_latent(
    transformer: torch.nn.Module,
    alphas: torch.Tensor,
    prompt_embed: torch.Tensor,
    prompt_mask: torch.Tensor,
    seed: int,
    steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, float, int]:
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(
        (1, *LATENT_SHAPE), generator=generator, device=device, dtype=torch.float16
    )
    calls = 0
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for start_index, target_index in phase_pairs(steps):
            start_t = state_timestep(start_index)
            target_t = state_timestep(target_index)
            output = transformer(
                latent,
                encoder_hidden_states=prompt_embed,
                encoder_attention_mask=prompt_mask,
                timestep=torch.full((1,), start_t, device=device, dtype=torch.long),
                added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                return_dict=False,
            )[0]
            calls += 1
            latent = deterministic_jump(
                latent,
                split_epsilon_prediction(output),
                start_t,
                target_t,
                alphas,
            ).to(dtype=torch.float16)
    torch.cuda.synchronize()
    return latent.cpu(), time.perf_counter() - started, calls


def generate_sets(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")
    if args.prompt_limit <= 0 or not 1 <= args.seeds_per_prompt <= 4:
        raise ValueError("Invalid prompt/seed evaluation limits.")
    teacher_manifest = json.loads(
        args.teacher_manifest.read_text(encoding="utf-8")
    )
    if teacher_manifest.get("status") != "PASS":
        raise ValueError("Teacher manifest is not PASS.")
    records = _load_records(args)
    prompt_features = _encode_unique_prompts(records, args)
    # Empty embeddings are only needed by a CFG teacher. Reuse the canonical
    # training cache when possible instead of retaining T5 beside PixArt.
    if float(teacher_manifest["teacher_guidance_scale"]) > 1.0:
        canonical_cache = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "features"
            / "t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt"
        )
        cache = torch.load(canonical_cache, map_location="cpu", weights_only=True)
        empty_embed = cache["empty_prompt_embeds"]
        empty_mask = cache["empty_prompt_attention_mask"]
    else:
        first_embed, first_mask = next(iter(prompt_features.values()))
        empty_embed = torch.zeros_like(first_embed)
        empty_mask = torch.zeros_like(first_mask)

    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        DPMSolverMultistepScheduler,
        PixArtTransformer2DModel,
    )
    from diffusers.image_processor import VaeImageProcessor
    from peft import PeftModel

    device = torch.device("cuda")
    teacher_base = PixArtTransformer2DModel.from_pretrained(
        args.transformer_model,
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    )
    teacher = PeftModel.from_pretrained(
        teacher_base,
        resolve_adapter_dir(teacher_manifest["adapter_dir"]),
        is_trainable=False,
    ).to(device).eval()
    teacher_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.component_model,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )
    teacher_results: list[tuple[dict[str, Any], torch.Tensor, float]] = []
    for record in records:
        embed, mask = prompt_features[record["prompt_id"]]
        torch.cuda.synchronize()
        started = time.perf_counter()
        latent = _generate_teacher_final(
            transformer=teacher,
            scheduler=teacher_scheduler,
            prompt_embed=embed.to(device),
            prompt_mask=mask.to(device),
            empty_embed=empty_embed.to(device),
            empty_mask=empty_mask.to(device),
            seed=record["seed"],
            guidance_scale=float(teacher_manifest["teacher_guidance_scale"]),
            device=device,
        )
        torch.cuda.synchronize()
        teacher_results.append(
            (record, latent, time.perf_counter() - started)
        )
    del teacher, teacher_base
    gc.collect()
    torch.cuda.empty_cache()

    student_base = PixArtTransformer2DModel.from_pretrained(
        args.transformer_model,
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    )
    student = PeftModel.from_pretrained(
        student_base,
        resolve_adapter_dir(args.student_adapter),
        is_trainable=False,
    ).to(device).eval()
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.component_model,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )
    alphas = noise_scheduler.alphas_cumprod.float()
    student_results: list[tuple[dict[str, Any], torch.Tensor, float, int]] = []
    for record in records:
        embed, mask = prompt_features[record["prompt_id"]]
        latent, seconds, calls = _student_latent(
            student,
            alphas,
            embed.to(device),
            mask.to(device),
            record["seed"],
            args.student_steps,
            device,
        )
        student_results.append((record, latent, seconds, calls))
    del student, student_base
    gc.collect()
    torch.cuda.empty_cache()

    vae = AutoencoderKL.from_pretrained(
        args.component_model,
        subfolder="vae",
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    processor = VaeImageProcessor(vae_scale_factor=8)
    teacher_dir = args.output_dir.resolve() / "teacher"
    student_dir = args.output_dir.resolve() / "student"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    student_dir.mkdir(parents=True, exist_ok=True)

    def decode_and_save(
        latent: torch.Tensor, path: Path, metadata: dict[str, Any]
    ) -> None:
        if path.is_file() and path.with_suffix(".json").is_file() and not args.overwrite:
            return
        with torch.inference_mode():
            decoded = vae.decode(
                latent.to(device) / vae.config.scaling_factor,
                return_dict=False,
            )[0]
        image = processor.postprocess(decoded, output_type="pil")[0]
        image.save(path)
        write_json(path.with_suffix(".json"), metadata)

    for record, latent, seconds in teacher_results:
        decode_and_save(
            latent,
            teacher_dir / record["filename"],
            {
                "status": "PASS",
                **record,
                "num_inference_steps": 20,
                "guidance_scale": teacher_manifest["teacher_guidance_scale"],
                "classifier_free_guidance_branch": (
                    teacher_manifest["teacher_guidance_scale"] > 1.0
                ),
                "transformer_forward_calls": 20,
                "denoise_seconds": seconds,
            },
        )
    for record, latent, seconds, calls in student_results:
        decode_and_save(
            latent,
            student_dir / record["filename"],
            {
                "status": "PASS",
                **record,
                "num_inference_steps": args.student_steps,
                "guidance_scale": 1.0,
                "classifier_free_guidance_branch": False,
                "transformer_forward_calls": calls,
                "denoise_seconds": seconds,
            },
        )
    result = {
        "status": "PASS",
        "teacher_id": teacher_manifest["teacher_id"],
        "student_adapter": str(resolve_adapter_dir(args.student_adapter)),
        "student_steps": args.student_steps,
        "prompt_count": args.prompt_limit,
        "seeds_per_prompt": args.seeds_per_prompt,
        "image_count_per_model": len(records),
        "teacher_images": str(teacher_dir),
        "student_images": str(student_dir),
    }
    write_json(args.output_dir / "generation_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generate_sets(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
