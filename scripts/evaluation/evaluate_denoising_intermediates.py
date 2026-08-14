#!/usr/bin/env python3
"""Evaluate every denoising call of the primary Teacher B/4-step/2-step models.

This is inference-only: it never updates or saves model parameters.  The fixed
eval-16 prompt and its four held-out evaluation seeds are read from the saved
T5 evaluation cache.  Every latent state after a Transformer call is decoded,
scored with CLIP, and written to CSV/JSON plus compact report figures.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.distillation.common import (  # noqa: E402
    COMPONENT_MODEL,
    LATENT_SHAPE,
    TEACHER_TIMESTEPS,
    TRANSFORMER_MODEL,
    deterministic_jump,
    phase_pairs,
    resolve_adapter_dir,
    split_epsilon_prediction,
    state_timestep,
    write_json,
)
from scripts.distillation.evaluation_prompt_cache import (  # noqa: E402
    DEFAULT_CACHE,
    load_evaluation_prompt_cache,
)
from scripts.distillation.evaluate_distilled_impl_clip_compat import (  # noqa: E402
    F_normalize,
)


DEFAULT_PROMPT_ID = "eval-16"
DEFAULT_SEEDS = (10161, 10162, 10163, 10164)
DEFAULT_PROMPT = (
    "A lone fishing boat crossing a vast river beneath high cliffs, "
    "traditional Chinese ink wash painting style, shuimo hua"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save and score every denoising call without retraining."
    )
    parser.add_argument("--prompt-id", default=DEFAULT_PROMPT_ID)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--teacher-adapter",
        type=Path,
        default=ROOT / "outputs" / "style_teacher" / "best_ink_wash_lora_plant209_step4000",
    )
    parser.add_argument(
        "--four-step-adapter",
        type=Path,
        default=ROOT
        / "outputs"
        / "distillation_experiments"
        / "teacher_b_extend6k_then2step"
        / "student_4step_extended_to6000"
        / "best_adapter",
    )
    parser.add_argument(
        "--two-step-adapter",
        type=Path,
        default=ROOT
        / "outputs"
        / "distillation_experiments"
        / "teacher_b_extend6k_then2step"
        / "student_2step_from_6000_4step"
        / "best_adapter",
    )
    parser.add_argument("--evaluation-prompt-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--empty-prompt-cache",
        type=Path,
        default=ROOT
        / "data"
        / "features"
        / "t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "evaluation" / "denoising_intermediates_eval16",
    )
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--transformer-model", default=TRANSFORMER_MODEL)
    parser.add_argument("--component-model", default=COMPONENT_MODEL)
    parser.add_argument("--teacher-guidance-scale", type=float, default=1.5)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def _empty_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _load_lora(adapter: Path, args: argparse.Namespace, device: torch.device):
    from diffusers import PixArtTransformer2DModel
    from peft import PeftModel

    base = PixArtTransformer2DModel.from_pretrained(
        args.transformer_model,
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    )
    model = PeftModel.from_pretrained(
        base, resolve_adapter_dir(adapter), is_trainable=False
    ).to(device).eval()
    return base, model


def _teacher_states(
    model: Any,
    scheduler: Any,
    prompt_embed: torch.Tensor,
    prompt_mask: torch.Tensor,
    empty_embed: torch.Tensor,
    empty_mask: torch.Tensor,
    seeds: Sequence[int],
    guidance_scale: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    embeds = torch.cat([empty_embed, prompt_embed])
    masks = torch.cat([empty_mask, prompt_mask])
    for seed in seeds:
        scheduler.set_timesteps(20, device=device)
        actual = tuple(int(value) for value in scheduler.timesteps.tolist())
        if actual != TEACHER_TIMESTEPS:
            raise RuntimeError(f"Teacher timestep schedule changed: {actual}")
        generator = torch.Generator(device=device).manual_seed(seed)
        latent = torch.randn(
            (1, *LATENT_SHAPE), generator=generator, device=device, dtype=torch.float16
        ) * scheduler.init_noise_sigma
        with torch.inference_mode():
            for call, timestep in enumerate(scheduler.timesteps, start=1):
                model_input = torch.cat([latent, latent])
                model_input = scheduler.scale_model_input(model_input, timestep)
                output = model(
                    model_input,
                    encoder_hidden_states=embeds,
                    encoder_attention_mask=masks,
                    timestep=timestep.reshape(1).expand(2),
                    added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                    return_dict=False,
                )[0]
                unconditional, conditional = output.chunk(2)
                guided = unconditional + guidance_scale * (conditional - unconditional)
                latent = scheduler.step(
                    split_epsilon_prediction(guided),
                    timestep,
                    latent,
                    return_dict=False,
                )[0]
                states.append(
                    {
                        "model": "20-step teacher",
                        "total_calls": 20,
                        "call": call,
                        "seed": seed,
                        "start_timestep": int(timestep.item()),
                        "target_timestep": (
                            int(scheduler.timesteps[call].item()) if call < 20 else -1
                        ),
                        "latent": latent.detach().to("cpu", dtype=torch.float16),
                    }
                )
    return states


def _student_states(
    model: Any,
    alphas: torch.Tensor,
    prompt_embed: torch.Tensor,
    prompt_mask: torch.Tensor,
    seeds: Sequence[int],
    total_calls: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        latent = torch.randn(
            (1, *LATENT_SHAPE), generator=generator, device=device, dtype=torch.float16
        )
        with torch.inference_mode():
            for call, (start_index, target_index) in enumerate(
                phase_pairs(total_calls), start=1
            ):
                start_t = state_timestep(start_index)
                target_t = state_timestep(target_index)
                timesteps = torch.full((1,), start_t, device=device, dtype=torch.long)
                output = model(
                    latent,
                    encoder_hidden_states=prompt_embed,
                    encoder_attention_mask=prompt_mask,
                    timestep=timesteps,
                    added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                    return_dict=False,
                )[0]
                latent = deterministic_jump(
                    latent,
                    split_epsilon_prediction(output),
                    start_t,
                    target_t,
                    alphas,
                ).to(dtype=torch.float16)
                states.append(
                    {
                        "model": f"{total_calls}-step student",
                        "total_calls": total_calls,
                        "call": call,
                        "seed": seed,
                        "start_timestep": start_t,
                        "target_timestep": target_t,
                        "latent": latent.detach().to("cpu", dtype=torch.float16),
                    }
                )
    return states


def _decode_states(
    states: list[dict[str, Any]], args: argparse.Namespace, device: torch.device
) -> None:
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor

    vae = AutoencoderKL.from_pretrained(
        args.component_model,
        subfolder="vae",
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    processor = VaeImageProcessor(vae_scale_factor=8)
    image_root = args.output_dir.resolve() / "images"
    with torch.inference_mode():
        for state in states:
            decoded = vae.decode(
                state["latent"].to(device) / vae.config.scaling_factor,
                return_dict=False,
            )[0]
            image = processor.postprocess(decoded, output_type="pil")[0]
            role = f"{state['total_calls']}step"
            path = image_root / role / f"seed{state['seed']}_call{state['call']:02d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            state["image"] = str(path)
            state.pop("latent")
    del vae
    _empty_cuda_cache()


def _score_states(
    states: list[dict[str, Any]], args: argparse.Namespace, device: torch.device
) -> None:
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(
        args.clip_model, local_files_only=args.local_files_only
    )
    model = CLIPModel.from_pretrained(
        args.clip_model, local_files_only=args.local_files_only
    ).to(device).eval()
    text_batch = processor(text=[args.prompt], return_tensors="pt")
    with torch.inference_mode():
        text_features = F_normalize(
            model.get_text_features(
                input_ids=text_batch.input_ids.to(device),
                attention_mask=text_batch.attention_mask.to(device),
            )
        )
    for start in range(0, len(states), 8):
        batch_states = states[start : start + 8]
        images: list[Image.Image] = []
        for state in batch_states:
            with Image.open(state["image"]) as image:
                images.append(image.convert("RGB").copy())
        batch = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            image_features = F_normalize(
                model.get_image_features(pixel_values=batch.pixel_values.to(device))
            )
        scores = (image_features * text_features).sum(dim=1).tolist()
        for state, score in zip(batch_states, scores, strict=True):
            state["clip_score"] = float(score)
    del model, processor
    _empty_cuda_cache()


def _write_statistics(states: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = (
        "model",
        "total_calls",
        "call",
        "seed",
        "start_timestep",
        "target_timestep",
        "clip_score",
        "image",
    )
    with (output_dir / "per_seed_intermediate_clip.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in states)

    summary: list[dict[str, Any]] = []
    keys = sorted({(row["total_calls"], row["call"]) for row in states}, reverse=True)
    for total_calls, call in keys:
        values = [
            row["clip_score"]
            for row in states
            if row["total_calls"] == total_calls and row["call"] == call
        ]
        summary.append(
            {
                "model": next(
                    row["model"]
                    for row in states
                    if row["total_calls"] == total_calls and row["call"] == call
                ),
                "total_calls": total_calls,
                "call": call,
                "n_seeds": len(values),
                "mean_clip": statistics.fmean(values),
                "sample_sd_clip": statistics.stdev(values),
                "sample_variance_clip": statistics.variance(values),
                "min_clip": min(values),
                "max_clip": max(values),
            }
        )
    with (output_dir / "intermediate_clip_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def _font(size: int, bold: bool = False):
    path = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _make_image_grid(states: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    selected_calls = {20: (1, 5, 10, 15, 20), 4: (1, 2, 3, 4), 2: (1, 2)}
    seed = 10163 if 10163 in args.seeds else args.seeds[0]
    rows: list[tuple[int, list[dict[str, Any]]]] = []
    for total in (20, 4, 2):
        rows.append(
            (
                total,
                [
                    next(
                        row
                        for row in states
                        if row["total_calls"] == total
                        and row["seed"] == seed
                        and row["call"] == call
                    )
                    for call in selected_calls[total]
                ],
            )
        )
    cell, gap, left, top, row_gap = 256, 12, 170, 64, 74
    width = left + 5 * cell + 4 * gap + 24
    height = top + 3 * cell + 2 * row_gap + 66
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, label_font, small_font = _font(25, True), _font(22, True), _font(17)
    draw.text((24, 18), f"Denoising intermediates: {args.prompt_id}, seed {seed}", fill="black", font=title_font)
    y = top
    for total, row_states in rows:
        draw.text((22, y + 96), f"{total}-step", fill="#17365D", font=label_font)
        for index, state in enumerate(row_states):
            x = left + index * (cell + gap)
            with Image.open(state["image"]) as source:
                image = source.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline="#17365D", width=2)
            label = f"call {state['call']}  CLIP {state['clip_score']:.3f}"
            draw.text((x + 4, y + cell + 7), label, fill="black", font=small_font)
        y += cell + row_gap
    output = args.output_dir.resolve() / "intermediate_images_eval16_seed10163.png"
    canvas.save(output)
    return output


def _make_curve(summary: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    import matplotlib.pyplot as plt

    colors = {20: "#1f77b4", 4: "#2ca02c", 2: "#d62728"}
    labels = {20: "20-step Teacher B", 4: "4-step student", 2: "2-step student"}
    fig, ax = plt.subplots(figsize=(7.2, 3.35), dpi=180)
    for total in (20, 4, 2):
        rows = sorted((row for row in summary if row["total_calls"] == total), key=lambda row: row["call"])
        x = [row["call"] for row in rows]
        y = [row["mean_clip"] for row in rows]
        sd = [row["sample_sd_clip"] for row in rows]
        ax.errorbar(x, y, yerr=sd, marker="o", linewidth=1.8, capsize=2.5, color=colors[total], label=labels[total])
    ax.set_xlabel("Transformer call completed")
    ax.set_ylabel("CLIP text-image alignment")
    ax.set_title(f"Intermediate alignment on {args.prompt_id} (mean +/- SD across 4 seeds)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output = args.output_dir.resolve() / "intermediate_clip_curve.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")
    if len(args.seeds) < 2 or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("At least two unique seeds are required.")
    record = {"prompt_id": args.prompt_id, "prompt": args.prompt}
    features = load_evaluation_prompt_cache(
        args.evaluation_prompt_cache, [record], component_model=args.component_model
    )
    prompt_embed, prompt_mask = features[args.prompt_id]
    empty_cache = torch.load(
        args.empty_prompt_cache.resolve(), map_location="cpu", weights_only=True
    )
    empty_embed = empty_cache["empty_prompt_embeds"]
    empty_mask = empty_cache["empty_prompt_attention_mask"]
    device = torch.device("cuda")
    prompt_embed, prompt_mask = prompt_embed.to(device), prompt_mask.to(device)
    empty_embed, empty_mask = empty_embed.to(device), empty_mask.to(device)

    from diffusers import DDPMScheduler, DPMSolverMultistepScheduler

    states: list[dict[str, Any]] = []
    base, model = _load_lora(args.teacher_adapter, args, device)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.component_model, subfolder="scheduler", local_files_only=args.local_files_only
    )
    states.extend(
        _teacher_states(
            model, scheduler, prompt_embed, prompt_mask, empty_embed, empty_mask,
            args.seeds, args.teacher_guidance_scale, device
        )
    )
    del model, base, scheduler
    _empty_cuda_cache()

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.component_model, subfolder="scheduler", local_files_only=args.local_files_only
    )
    alphas = noise_scheduler.alphas_cumprod.float()
    for total, adapter in ((4, args.four_step_adapter), (2, args.two_step_adapter)):
        base, model = _load_lora(adapter, args, device)
        states.extend(
            _student_states(
                model, alphas, prompt_embed, prompt_mask, args.seeds, total, device
            )
        )
        del model, base
        _empty_cuda_cache()
    del noise_scheduler
    _empty_cuda_cache()

    _decode_states(states, args, device)
    _score_states(states, args, device)
    summary = _write_statistics(states, args)
    image_grid = _make_image_grid(states, args)
    curve = _make_curve(summary, args)
    result = {
        "format_version": 1,
        "status": "PASS",
        "training_performed": False,
        "prompt_id": args.prompt_id,
        "prompt": args.prompt,
        "seeds": list(args.seeds),
        "seed_count": len(args.seeds),
        "models": ["20-step teacher", "4-step student", "2-step student"],
        "state_count": len(states),
        "clip_model": args.clip_model,
        "student_guidance_scale": 1.0,
        "teacher_guidance_scale": args.teacher_guidance_scale,
        "per_seed_csv": str(args.output_dir.resolve() / "per_seed_intermediate_clip.csv"),
        "summary_csv": str(args.output_dir.resolve() / "intermediate_clip_summary.csv"),
        "image_grid": str(image_grid),
        "curve": str(curve),
    }
    write_json(args.output_dir.resolve() / "evaluation_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    evaluate(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
