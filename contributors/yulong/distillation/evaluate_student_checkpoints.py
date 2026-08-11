#!/usr/bin/env python3
"""Evaluate PixArt 20->10 distillation across prompts, seeds and checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler, PixArtSigmaPipeline, PixArtTransformer2DModel
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, CLIPModel

from common import COMPONENT_MODEL, TRANSFORMER_MODEL, model_snapshot_source, resolve_adapter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "precomputed_prompts" / "focused_evaluation_prompts.pt"
DEFAULT_TEACHER = (
    PROJECT_ROOT
    / "models"
    / "lora_training_512"
    / "style_teacher_r16_lr1e-5_bs1_steps10000_seed42"
    / "checkpoints"
    / "step_004000"
)
DEFAULT_STUDENT_RUN = (
    PROJECT_ROOT
    / "models"
    / "distilled_students"
    / "student_20to10_teacher_step4000_g2"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "distillation"
    / "student_20to10_g2"
    / "evaluation_8prompts_3seeds"
)
STYLE_PREFIXES = (
    r"^traditional Chinese ink wash painting style\s+sumi-e style,\s*",
    r"^Chinese ink wash painting style,\s*Sumi-e,\s*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument(
        "--reference-adapter",
        type=Path,
        help="Optional earlier-stage reference, such as the original 20-step Teacher.",
    )
    parser.add_argument("--teacher-adapter", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--student-run", type=Path, default=DEFAULT_STUDENT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-steps", type=int, default=20)
    parser.add_argument("--reference-guidance", type=float, default=2.0)
    parser.add_argument("--teacher-steps", type=int, default=20)
    parser.add_argument("--teacher-guidance", type=float, default=2.0)
    parser.add_argument("--student-steps", type=int, default=10)
    parser.add_argument("--student-guidance", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 123, 2026))
    parser.add_argument(
        "--models",
        nargs="+",
        default=("teacher", "4000", "8000", "12000", "16000"),
        help="Reference, teacher and/or any positive numeric Student checkpoint steps.",
    )
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    return parser.parse_args()


def content_only_prompt(prompt: str) -> str:
    result = prompt.strip()
    for pattern in STYLE_PREFIXES:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    return result


def model_specs(args: argparse.Namespace) -> list[dict]:
    specs = []
    for name in args.models:
        if name == "reference":
            if args.reference_adapter is None:
                raise ValueError("--models reference requires --reference-adapter")
            specs.append(
                {
                    "key": "reference",
                    "label": f"Reference {args.reference_steps}-step",
                    "adapter": resolve_adapter(args.reference_adapter),
                    "kind": "reference",
                    "optimizer_step": 0,
                    "sort_order": -2,
                    "inference_steps": args.reference_steps,
                    "guidance": args.reference_guidance,
                }
            )
        elif name == "teacher":
            specs.append(
                {
                    "key": "teacher",
                    "label": f"Teacher {args.teacher_steps}-step",
                    "adapter": resolve_adapter(args.teacher_adapter),
                    "kind": "teacher",
                    "optimizer_step": 0,
                    "sort_order": -1,
                    "inference_steps": args.teacher_steps,
                    "guidance": args.teacher_guidance,
                }
            )
        else:
            try:
                step = int(name)
            except ValueError as error:
                raise ValueError(
                    f"Unsupported --models value {name!r}; use reference, teacher, or a positive step."
                ) from error
            if step < 1:
                raise ValueError(f"Student checkpoint step must be positive: {step}")
            adapter = resolve_adapter(
                args.student_run / "checkpoints" / f"step_{step:06d}"
            )
            specs.append(
                {
                    "key": f"student_{step:06d}",
                    "label": f"Student {step}-update",
                    "adapter": adapter,
                    "kind": "student",
                    "optimizer_step": step,
                    "sort_order": step,
                    "inference_steps": args.student_steps,
                    "guidance": args.student_guidance,
                }
            )
    return specs


def load_bundle(path: Path, max_prompts: int | None) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format") != "pixart_sigma_precomputed_prompt_embeddings_v1":
        raise ValueError("Incompatible prompt embedding bundle")
    count = len(bundle["prompts"])
    if max_prompts is not None:
        if not 1 <= max_prompts <= count:
            raise ValueError(f"--max-prompts must be 1..{count}")
        count = max_prompts
    bundle["path"] = path
    bundle["evaluation_count"] = count
    return bundle


def build_pipeline(adapter: Path) -> PixArtSigmaPipeline:
    transformer = PixArtTransformer2DModel.from_pretrained(
        model_snapshot_source(TRANSFORMER_MODEL),
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
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
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_one(
    pipe: PixArtSigmaPipeline,
    bundle: dict,
    prompt_index: int,
    seed: int,
    steps: int,
    guidance: float,
) -> tuple[Image.Image, float]:
    prompt_embeds = bundle["prompt_embeds"][prompt_index : prompt_index + 1].to(
        "cuda", dtype=torch.float16
    )
    prompt_mask = bundle["prompt_attention_masks"][prompt_index : prompt_index + 1].to(
        "cuda"
    )
    call_args = {
        "prompt": None,
        "prompt_embeds": prompt_embeds,
        "prompt_attention_mask": prompt_mask,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "height": 512,
        "width": 512,
        "generator": torch.Generator(device="cuda").manual_seed(seed),
        "use_resolution_binning": False,
    }
    if guidance > 1.0:
        call_args["negative_prompt"] = None
        call_args["negative_prompt_embeds"] = bundle["empty_prompt_embeds"].to(
            "cuda", dtype=torch.float16
        )
        call_args["negative_prompt_attention_mask"] = bundle[
            "empty_prompt_attention_mask"
        ].to("cuda")
    started = time.perf_counter()
    with torch.inference_mode():
        image = pipe(**call_args).images[0]
    torch.cuda.synchronize()
    return image, time.perf_counter() - started


def read_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def generate_images(
    args: argparse.Namespace, bundle: dict, specs: list[dict], output: Path
) -> list[dict]:
    records_path = output / "generation_records.jsonl"
    records = read_records(records_path)
    existing = {
        (row["model_key"], row["prompt_index"], row["seed"]): row for row in records
    }
    expected_keys = {
        (spec["key"], prompt_index, seed)
        for spec in specs
        for prompt_index in range(bundle["evaluation_count"])
        for seed in args.seeds
    }
    if args.skip_generation:
        missing = expected_keys.difference(existing)
        if missing:
            raise FileNotFoundError(
                f"--skip-generation requested but {len(missing)} records are missing"
            )
        return [existing[key] for key in sorted(expected_keys)]

    total = len(expected_keys)
    completed = len(expected_keys.intersection(existing))
    torch.backends.cuda.matmul.allow_tf32 = True
    for spec in specs:
        missing_for_model = [
            (prompt_index, seed)
            for prompt_index in range(bundle["evaluation_count"])
            for seed in args.seeds
            if (spec["key"], prompt_index, seed) not in existing
        ]
        if not missing_for_model:
            print(f"Skipping complete model: {spec['label']}")
            continue
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"Loading {spec['label']} ({len(missing_for_model)} images)...")
        pipe = build_pipeline(spec["adapter"])
        # Warm CUDA kernels and VAE once; this image is intentionally discarded.
        generate_one(
            pipe,
            bundle,
            prompt_index=0,
            seed=987654,
            steps=spec["inference_steps"],
            guidance=spec["guidance"],
        )
        for prompt_index, seed in missing_for_model:
            image, elapsed = generate_one(
                pipe,
                bundle,
                prompt_index,
                seed,
                spec["inference_steps"],
                spec["guidance"],
            )
            image_dir = output / "images" / spec["key"]
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"prompt{prompt_index:02d}_seed{seed}.png"
            image.save(image_path)
            record = {
                "model_key": spec["key"],
                "model_label": spec["label"],
                "model_kind": spec["kind"],
                "optimizer_step": spec["optimizer_step"],
                "sort_order": spec["sort_order"],
                "adapter": str(spec["adapter"]),
                "inference_steps": spec["inference_steps"],
                "guidance": spec["guidance"],
                "scheduler": "DDIMScheduler",
                "timestep_spacing": "trailing",
                "prompt_index": prompt_index,
                "prompt": bundle["prompts"][prompt_index],
                "content_prompt": content_only_prompt(
                    bundle["prompts"][prompt_index]
                ),
                "seed": seed,
                "image_path": str(image_path.resolve()),
                "elapsed_seconds": elapsed,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing[(spec["key"], prompt_index, seed)] = record
            completed += 1
            print(
                f"Generated {completed}/{total}: {spec['key']} "
                f"prompt={prompt_index} seed={seed} ({elapsed:.2f}s)"
            )
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    return [existing[key] for key in sorted(expected_keys)]


@torch.inference_mode()
def add_clip_metrics(rows: list[dict], model_id: str, batch_size: int) -> None:
    source = model_snapshot_source(model_id)
    model = CLIPModel.from_pretrained(
        source, torch_dtype=torch.float16, attn_implementation="eager"
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(source)

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = []
        for row in batch:
            with Image.open(row["image_path"]) as image:
                images.append(image.convert("RGB"))
        image_inputs = processor(images=images, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to("cuda")
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        image_features = model.visual_projection(vision_outputs.pooler_output)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        for row, feature in zip(batch, image_features.float().cpu(), strict=True):
            row["clip_image_feature"] = feature.tolist()

        for field, output in (
            ("prompt", "clip_full_cosine"),
            ("content_prompt", "clip_content_cosine"),
        ):
            text_inputs = processor(
                text=[row[field] for row in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = {key: value.to("cuda") for key, value in text_inputs.items()}
            text_outputs = model.text_model(**text_inputs)
            text_features = model.text_projection(text_outputs.pooler_output)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarities = (
                image_features * text_features
            ).sum(dim=-1).float().cpu().tolist()
            for row, similarity in zip(batch, similarities, strict=True):
                row[output] = similarity
        print(f"CLIP scored {min(start + batch_size, len(rows))}/{len(rows)}")
    del model
    gc.collect()
    torch.cuda.empty_cache()


def add_teacher_reference_metrics(rows: list[dict]) -> None:
    teacher_rows = {
        (row["prompt_index"], row["seed"]): row
        for row in rows
        if row["model_kind"] == "teacher"
    }
    if not teacher_rows:
        raise ValueError("Teacher records are required for reference metrics")
    reference_rows = {
        (row["prompt_index"], row["seed"]): row
        for row in rows
        if row["model_kind"] == "reference"
    }
    features = {
        (row["model_key"], row["prompt_index"], row["seed"]): np.asarray(
            row["clip_image_feature"], dtype=np.float32
        )
        for row in rows
    }
    for row in rows:
        teacher = teacher_rows[(row["prompt_index"], row["seed"])]
        student_feature = features[
            (row["model_key"], row["prompt_index"], row["seed"])
        ]
        teacher_feature = features[
            (teacher["model_key"], teacher["prompt_index"], teacher["seed"])
        ]
        row["clip_teacher_image_cosine"] = float(
            np.dot(student_feature, teacher_feature)
        )
        with Image.open(row["image_path"]) as image:
            current = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(teacher["image_path"]) as image:
            reference = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        difference = current - reference
        mae = float(np.abs(difference).mean())
        mse = float(np.square(difference).mean())
        row["pixel_mae"] = mae
        row["pixel_psnr_db"] = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
        if reference_rows:
            reference_row = reference_rows[(row["prompt_index"], row["seed"])]
            reference_feature = features[
                (
                    reference_row["model_key"],
                    reference_row["prompt_index"],
                    reference_row["seed"],
                )
            ]
            row["clip_reference_image_cosine"] = float(
                np.dot(student_feature, reference_feature)
            )
            with Image.open(reference_row["image_path"]) as image:
                earlier_reference = (
                    np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                )
            reference_difference = current - earlier_reference
            reference_mae = float(np.abs(reference_difference).mean())
            reference_mse = float(np.square(reference_difference).mean())
            row["pixel_reference_mae"] = reference_mae
            row["pixel_reference_psnr_db"] = (
                float("inf")
                if reference_mse == 0
                else 10.0 * math.log10(1.0 / reference_mse)
            )
    for row in rows:
        row.pop("clip_image_feature", None)


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["model_key"]].append(row)
    summary = []
    for key, group in sorted(
        groups.items(), key=lambda item: item[1][0].get("sort_order", 0)
    ):
        first = group[0]
        values = {}
        for metric in (
            "elapsed_seconds",
            "clip_full_cosine",
            "clip_content_cosine",
            "clip_teacher_image_cosine",
            "pixel_mae",
        ):
            values[f"{metric}_mean"], values[f"{metric}_std"] = mean_std(
                [float(row[metric]) for row in group]
            )
        if "clip_reference_image_cosine" in first:
            for metric in (
                "clip_reference_image_cosine",
                "pixel_reference_mae",
            ):
                values[f"{metric}_mean"], values[f"{metric}_std"] = mean_std(
                    [float(row[metric]) for row in group]
                )
        elapsed_values = [float(row["elapsed_seconds"]) for row in group]
        values["elapsed_seconds_median"] = statistics.median(elapsed_values)
        values["elapsed_seconds_p25"] = float(np.percentile(elapsed_values, 25))
        values["elapsed_seconds_p75"] = float(np.percentile(elapsed_values, 75))
        psnr = [float(row["pixel_psnr_db"]) for row in group if math.isfinite(float(row["pixel_psnr_db"]))]
        values["pixel_psnr_db_mean"] = statistics.mean(psnr) if psnr else math.inf
        values["pixel_psnr_db_std"] = statistics.stdev(psnr) if len(psnr) > 1 else 0.0
        summary.append(
            {
                "model_key": key,
                "model_label": first["model_label"],
                "model_kind": first["model_kind"],
                "optimizer_step": first["optimizer_step"],
                "sort_order": first.get("sort_order", 0),
                "inference_steps": first["inference_steps"],
                "guidance": first["guidance"],
                "n": len(group),
                **values,
            }
        )
    teacher_time = next(
        row["elapsed_seconds_mean"] for row in summary if row["model_kind"] == "teacher"
    )
    teacher_median_time = next(
        row["elapsed_seconds_median"]
        for row in summary
        if row["model_kind"] == "teacher"
    )
    for row in summary:
        row["speedup_vs_teacher"] = teacher_time / row["elapsed_seconds_mean"]
        row["speedup_vs_teacher_median"] = (
            teacher_median_time / row["elapsed_seconds_median"]
        )
    reference_summary = next(
        (row for row in summary if row["model_kind"] == "reference"), None
    )
    if reference_summary is not None:
        for row in summary:
            row["speedup_vs_reference"] = (
                reference_summary["elapsed_seconds_mean"]
                / row["elapsed_seconds_mean"]
            )
            row["speedup_vs_reference_median"] = (
                reference_summary["elapsed_seconds_median"]
                / row["elapsed_seconds_median"]
            )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheets(rows: list[dict], output: Path) -> None:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["prompt_index"], row["seed"])].append(row)
    contact_dir = output / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for (prompt_index, seed), group in sorted(grouped.items()):
        group.sort(key=lambda row: row.get("sort_order", 0))
        images = [Image.open(row["image_path"]).convert("RGB") for row in group]
        label_height = 34
        sheet = Image.new(
            "RGB", (512 * len(images), 512 + label_height), "white"
        )
        draw = ImageDraw.Draw(sheet)
        for column, (row, image) in enumerate(zip(group, images, strict=True)):
            x = column * 512
            draw.text((x + 7, 10), row["model_label"], fill="black", font=font)
            sheet.paste(image, (x, label_height))
        sheet.save(contact_dir / f"prompt{prompt_index:02d}_seed{seed}.png")
        for image in images:
            image.close()


def plot_summary(summary: list[dict], output: Path) -> None:
    labels = [row["model_label"].replace("-update", "") for row in summary]
    x = np.arange(len(summary))
    blue = "#2F6B9A"
    gold = "#D89B2B"
    charcoal = "#252A30"
    grid = "#D9DEE3"
    n_cases = summary[0]["n"]

    def finish(path: Path) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close()

    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor="white")
    means = [row["clip_teacher_image_cosine_mean"] for row in summary]
    stds = [row["clip_teacher_image_cosine_std"] for row in summary]
    ax.errorbar(x, means, yerr=stds, color=blue, marker="o", linewidth=2, capsize=4)
    ax.set_title("Teacher–Student CLIP Image Similarity")
    ax.set_ylabel(f"Cosine similarity (mean ± SD, {n_cases} matched cases)")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=charcoal)
    finish(output / "chart_teacher_similarity.png")

    if "clip_reference_image_cosine_mean" in summary[0]:
        fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor="white")
        teacher_similarity = [
            row["clip_teacher_image_cosine_mean"] for row in summary
        ]
        reference_similarity = [
            row["clip_reference_image_cosine_mean"] for row in summary
        ]
        ax.plot(
            x,
            teacher_similarity,
            color=blue,
            marker="o",
            linewidth=2,
            label="Similarity to current Teacher",
        )
        ax.plot(
            x,
            reference_similarity,
            color=gold,
            marker="s",
            linewidth=2,
            label="Similarity to earlier reference",
        )
        ax.set_title("Progressive Distillation Reference Similarity")
        ax.set_ylabel(f"CLIP image cosine (mean across {n_cases} cases)")
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        finish(output / "chart_dual_reference_similarity.png")

    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor="white")
    full = [row["clip_full_cosine_mean"] for row in summary]
    content = [row["clip_content_cosine_mean"] for row in summary]
    ax.plot(x, full, color=blue, marker="o", linewidth=2, label="Full prompt")
    ax.plot(x, content, color=gold, marker="s", linewidth=2, label="Content-only prompt")
    ax.set_title("CLIP Text–Image Alignment")
    ax.set_ylabel(f"Cosine similarity (mean across {n_cases} images)")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    finish(output / "chart_prompt_alignment.png")

    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor="white")
    times = [row["elapsed_seconds_median"] for row in summary]
    lower = [
        row["elapsed_seconds_median"] - row["elapsed_seconds_p25"]
        for row in summary
    ]
    upper = [
        row["elapsed_seconds_p75"] - row["elapsed_seconds_median"]
        for row in summary
    ]
    colors = [
        charcoal
        if row["model_kind"] == "reference"
        else blue
        if row["model_kind"] == "teacher"
        else gold
        for row in summary
    ]
    bars = ax.bar(
        x,
        times,
        yerr=np.asarray([lower, upper]),
        capsize=4,
        color=colors,
        edgecolor=charcoal,
        linewidth=0.7,
    )
    ax.bar_label(bars, fmt="%.2fs", padding=3)
    ax.set_title("Median Generation Time")
    ax.set_ylabel(f"Seconds per 512×512 image (median and IQR, n={n_cases})")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylim(0, max(times) * 1.2)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    finish(output / "chart_generation_time.png")


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() and not args.plan_only:
        raise RuntimeError("CUDA is required")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.student_run = args.student_run.expanduser().resolve()
    bundle = load_bundle(args.embeddings, args.max_prompts)
    specs = model_specs(args)
    expected = bundle["evaluation_count"] * len(args.seeds) * len(specs)
    print("DISTILLATION EVALUATION PLAN")
    print(f"Prompts : {bundle['evaluation_count']}")
    print(f"Seeds   : {list(args.seeds)}")
    print(f"Models  : {[spec['key'] for spec in specs]}")
    print(f"Images  : {expected}")
    print(f"Output  : {args.output_dir}")
    if args.plan_only:
        print("PLAN-ONLY PASSED")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "analytical_question": (
            f"Which {args.student_steps}-step Student checkpoint best matches the "
            f"{args.teacher_steps}-step Teacher, remains faithful to the earlier "
            f"{args.reference_steps}-step reference, and reduces latency?"
        ),
        "prompts": bundle["evaluation_count"],
        "seeds": list(args.seeds),
        "models": [
            {**spec, "adapter": str(spec["adapter"])} for spec in specs
        ],
        "expected_images": expected,
        "primary_metrics": [
            "clip_teacher_image_cosine",
            "clip_reference_image_cosine",
        ],
        "guardrails": [
            "clip_content_cosine",
            "pixel_mae",
            "pixel_reference_mae",
            "visual deformation review",
        ],
        "timing_metric": "warmed single-image generation elapsed_seconds",
        "chart_map": [
            "two-series line: model vs current-Teacher and earlier-reference CLIP similarity",
            "two-series line: full/content prompt CLIP alignment",
            "bar: mean generation seconds by model",
        ],
    }
    (args.output_dir / "evaluation_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = generate_images(args, bundle, specs, args.output_dir)
    add_clip_metrics(rows, args.clip_model, args.clip_batch_size)
    add_teacher_reference_metrics(rows)
    summary = summarize(rows)
    write_csv(args.output_dir / "metrics_detailed.csv", rows)
    write_csv(args.output_dir / "metrics_summary.csv", summary)
    make_contact_sheets(rows, args.output_dir)
    plot_summary(summary, args.output_dir)
    print("EVALUATION COMPLETE")
    print(f"Detailed metrics: {args.output_dir / 'metrics_detailed.csv'}")
    print(f"Summary metrics : {args.output_dir / 'metrics_summary.csv'}")
    for row in summary:
        reference_text = ""
        if "clip_reference_image_cosine_mean" in row:
            reference_text = (
                f", reference_sim={row['clip_reference_image_cosine_mean']:.4f}, "
                f"reference_speedup={row['speedup_vs_reference_median']:.2f}x"
            )
        print(
            f"{row['model_key']:>12}: teacher_sim="
            f"{row['clip_teacher_image_cosine_mean']:.4f}, "
            f"content_clip={row['clip_content_cosine_mean']:.4f}, "
            f"median_time={row['elapsed_seconds_median']:.2f}s, "
            f"teacher_speedup={row['speedup_vs_teacher_median']:.2f}x"
            f"{reference_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
