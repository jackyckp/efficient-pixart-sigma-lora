#!/usr/bin/env python3
"""Comprehensive 30-Prompt Benchmark & CMMD Evaluation Script.

Uses exact trigger word: 'traditional Chinese ink wash painting style, shuimo hua'

Evaluates best dataset checkpoints across 4 traditional ink-wash prompt categories:
1. Landscapes 
2. Flora & Fauna 
3. Minimalist Composition 
4. Detailed Figures & Architecture 

Calculates both CLIPScore (Text-Image Alignment) and CMMD (CLIP Maximum Mean Discrepancy against ground-truth ink wash images).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image


TRIGGER_WORD = "traditional Chinese ink wash painting style, shuimo hua"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


VALIDATION_PROMPTS = [
    # Category 1: Landscapes 
    {"id": 1, "category": "Landscapes", "prompt": f"Misty mountain peaks enveloped in soft clouds, ancient pine tree on a cliff, {TRIGGER_WORD}"},
    {"id": 2, "category": "Landscapes", "prompt": f"Winding river flowing through steep mountain gorges, distant waterfall, {TRIGGER_WORD}"},
    {"id": 3, "category": "Landscapes", "prompt": f"Cascading waterfall plunging into a misty ravine, jagged rock formations, {TRIGGER_WORD}"},
    {"id": 4, "category": "Landscapes", "prompt": f"Snow-covered mountain range in winter, bare trees, frozen lake, {TRIGGER_WORD}"},
    {"id": 5, "category": "Landscapes", "prompt": f"Autumn mountains with sparse foliage, winding stone path leading to a ridge, {TRIGGER_WORD}"},
    {"id": 6, "category": "Landscapes", "prompt": f"Sunrise over sea of clouds and mountain spires, high contrast black ink brushwork, white space, {TRIGGER_WORD}"},
    {"id": 7, "category": "Landscapes", "prompt": f"Quiet lake reflecting towering mountain shadows, serene water surface, {TRIGGER_WORD}"},
    {"id": 8, "category": "Landscapes", "prompt": f"Storm clouds gathering above rugged cliffside pines, dynamic black ink splash technique, {TRIGGER_WORD}"},

    # Category 2: Flora & Fauna 
    {"id": 9, "category": "Flora_Fauna", "prompt": f"Ink wash bamboo in the wind, wet brush technique, delicate leaves, subtle grey tones, {TRIGGER_WORD}"},
    {"id": 10, "category": "Flora_Fauna", "prompt": f"A pair of flying cranes soaring above misty clouds, elegant brushstrokes, {TRIGGER_WORD}"},
    {"id": 11, "category": "Flora_Fauna", "prompt": f"Blooming plum blossoms on a gnarled branch, delicate ink wash gradients, soft grey background, {TRIGGER_WORD}"},
    {"id": 12, "category": "Flora_Fauna", "prompt": f"Solitary eagle perched on an ancient pine branch, sharp gaze, bold black ink brushwork, {TRIGGER_WORD}"},
    {"id": 13, "category": "Flora_Fauna", "prompt": f"Lotus flowers blooming in a quiet pond, large wet ink leaves, dragonfly hovering, {TRIGGER_WORD}"},
    {"id": 14, "category": "Flora_Fauna", "prompt": f"A wild horse galloping across an open plain, dynamic ink wash style, fluid brush lines, {TRIGGER_WORD}"},
    {"id": 15, "category": "Flora_Fauna", "prompt": f"Wild orchids clinging to a mossy cliff, graceful curved leaves, minimalist ink wash style, {TRIGGER_WORD}"},
    {"id": 16, "category": "Flora_Fauna", "prompt": f"Koi fish swimming in clear water, soft ink wash ripples, transparent ink gradients, {TRIGGER_WORD}"},

    # Category 3: Minimalist Composition 
    {"id": 17, "category": "Minimalist", "prompt": f"A single small boat on a vast calm lake, minimalist composition, wide white space, {TRIGGER_WORD}"},
    {"id": 18, "category": "Minimalist", "prompt": f"Solitary fisherman sitting on a riverbank with a fishing rod, vast empty background, {TRIGGER_WORD}"},
    {"id": 19, "category": "Minimalist", "prompt": f"Single bamboo stalk in the corner of a blank paper canvas, elegant white space composition, {TRIGGER_WORD}"},
    {"id": 20, "category": "Minimalist", "prompt": f"A lone pine tree silhouette against a faint crescent moon, subtle grey wash, high negative space, {TRIGGER_WORD}"},
    {"id": 21, "category": "Minimalist", "prompt": f"Faint outline of a distant mountain peak in heavy fog, minimalist ink wash composition, wide white space, {TRIGGER_WORD}"},
    {"id": 22, "category": "Minimalist", "prompt": f"A single falling leaf landing on still water, delicate ink ripple lines, minimalist composition, {TRIGGER_WORD}"},
    {"id": 23, "category": "Minimalist", "prompt": f"Distant flight of birds vanishing into empty mist, minimalist composition, wide negative space, {TRIGGER_WORD}"},

    # Category 4: Detailed Figures & Architecture 
    {"id": 24, "category": "Architecture", "prompt": f"Ancient wooden pavilion surrounded by swirling mountain fog, {TRIGGER_WORD}"},
    {"id": 25, "category": "Architecture", "prompt": f"Ancient scholar walking along a winding stone path, traditional robes, {TRIGGER_WORD}"},
    {"id": 26, "category": "Architecture", "prompt": f"Secluded stone temple tucked in a deep pine forest, mist rising, detailed architecture, {TRIGGER_WORD}"},
    {"id": 27, "category": "Architecture", "prompt": f"Traditional thatched cottage near a bamboo grove, flowing stream, {TRIGGER_WORD}"},
    {"id": 28, "category": "Architecture", "prompt": f"Ancient stone bridge spanning a misty river, small pavilion on a cliff, {TRIGGER_WORD}"},
    {"id": 29, "category": "Architecture", "prompt": f"Old scholar sitting inside a pavilion reading a book, mountain view, detailed ink wash technique, {TRIGGER_WORD}"},
    {"id": 30, "category": "Architecture", "prompt": f"Winding mountain staircase leading to a cloud-wrapped pagoda, {TRIGGER_WORD}"},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="30-Prompt Ink Wash Benchmark & CMMD Evaluation.")
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def compute_cmmd(gen_embeds: torch.Tensor, ref_embeds: torch.Tensor, sigma: float = 10.0) -> float:
    gen_norm = torch.nn.functional.normalize(gen_embeds, dim=-1)
    ref_norm = torch.nn.functional.normalize(ref_embeds, dim=-1)

    gamma = 1.0 / (2.0 * (sigma ** 2))
    dist_xx = torch.cdist(gen_norm, gen_norm, p=2) ** 2
    dist_yy = torch.cdist(ref_norm, ref_norm, p=2) ** 2
    dist_xy = torch.cdist(gen_norm, ref_norm, p=2) ** 2

    k_xx = torch.exp(-gamma * dist_xx).mean()
    k_yy = torch.exp(-gamma * dist_yy).mean()
    k_xy = torch.exp(-gamma * dist_xy).mean()

    mmd2 = k_xx + k_yy - 2.0 * k_xy
    return float(torch.clamp(mmd2, min=0.0).item())


def extract_reference_embeddings(clip_model, clip_processor, root_dir: Path) -> torch.Tensor:
    print("Extracting CLIP embeddings for reference ground-truth ink wash images...")
    ref_zip = root_dir / "data" / "archives" / "ink.zip"
    ref_embeds = []

    with zipfile.ZipFile(ref_zip) as archive:
        image_names = [n for n in archive.namelist() if n.lower().endswith((".jpg", ".png", ".jpeg"))][:100]
        for name in image_names:
            img_data = archive.read(name)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            inputs = clip_processor(images=img, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = clip_model.get_image_features(**inputs)
                emb = getattr(out, "pooler_output", out) if not isinstance(out, torch.Tensor) else out
                ref_embeds.append(emb.cpu())

    return torch.cat(ref_embeds, dim=0)


def main():
    args = build_parser().parse_args()
    root_dir = repository_root()
    output_root = args.output_root or (root_dir / "outputs" / "benchmark_30prompts")
    output_root.mkdir(parents=True, exist_ok=True)

    from transformers import CLIPModel, CLIPProcessor
    print("\n=== Loading CLIP Model for Evaluation ===")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to("cuda").eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    ref_embeds = extract_reference_embeddings(clip_model, clip_processor, root_dir).to("cuda")

    best_models = [
        {"model_id": "baseline", "name": "Baseline Model (Step 0)", "adapter": None},
        {"model_id": "n50_best", "name": "Best n50 (Step 1000)", "adapter": root_dir / "outputs" / "experiment_10k" / "r16_n50_steps10000" / "checkpoint-1000" / "lora_adapter"},
        {"model_id": "n100_best", "name": "Best n100 (Step 1000)", "adapter": root_dir / "outputs" / "experiment_10k" / "r16_n100_steps10000" / "checkpoint-1000" / "lora_adapter"},
        {"model_id": "plant209_best", "name": "Best plant209 (Step 4000)", "adapter": root_dir / "outputs" / "experiment_10k" / "r16_plant209_steps10000" / "checkpoint-4000" / "lora_adapter"},
        {"model_id": "n260_best", "name": "Best n260 (Step 2000)", "adapter": root_dir / "outputs" / "experiment_10k" / "r16_n260_steps10000" / "checkpoint-2000" / "lora_adapter"},
    ]

    generate_script = root_dir / "scripts" / "inference" / "generate_with_prompt.py"
    all_results = []
    cmmd_summary = []

    for m in best_models:
        model_id = m["model_id"]
        model_name = m["name"]
        adapter_path = m["adapter"]
        print(f"\n=======================================================")
        print(f"  Evaluating Model: {model_name}")
        print(f"=======================================================")

        gen_embeds_list = []
        model_clip_scores = []

        for item in VALIDATION_PROMPTS:
            p_id = item["id"]
            cat = item["category"]
            prompt = item["prompt"]

            img_out = output_root / "generations" / model_id / f"prompt_{p_id:02d}.png"
            img_out.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable, str(generate_script),
                "--prompt", prompt,
                "--output", str(img_out),
                "--seed", str(args.seed),
                "--num-inference-steps", "20",
                "--guidance-scale", str(args.guidance_scale),
                "--allow-seen-prompt",
            ]
            if adapter_path:
                cmd.extend(["--adapter", str(adapter_path)])
            else:
                cmd.append("--no-adapter")

            t0 = time.perf_counter()
            res = subprocess.run(cmd, capture_output=True, text=True)
            latency = time.perf_counter() - t0

            if res.returncode != 0:
                print(f"Generation error prompt {p_id}: {res.stderr}")
                continue

            # Compute CLIP embeddings and alignment score
            img = Image.open(img_out).convert("RGB")
            inputs = clip_processor(text=[prompt], images=img, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = clip_model(**inputs)
                clip_score = out.logits_per_image.item() / 100.0
                out_feat = clip_model.get_image_features(inputs.pixel_values)
                img_emb = getattr(out_feat, "pooler_output", out_feat) if not isinstance(out_feat, torch.Tensor) else out_feat
                gen_embeds_list.append(img_emb.cpu())

            model_clip_scores.append(clip_score)
            all_results.append({
                "model_id": model_id,
                "model_name": model_name,
                "prompt_id": p_id,
                "category": cat,
                "prompt": prompt,
                "clip_score": round(clip_score, 6),
                "latency_sec": round(latency, 4),
                "image_path": str(img_out),
            })
            print(f"  [{cat:12s}] P{p_id:02d} | CLIPScore: {clip_score:.4f} | Latency: {latency:.2f}s")

        # Compute CMMD for this model
        gen_embeds = torch.cat(gen_embeds_list, dim=0).to("cuda")
        cmmd_val = compute_cmmd(gen_embeds, ref_embeds)
        mean_clip = float(torch.tensor(model_clip_scores).mean().item())

        print(f"\n>>> Model '{model_name}' Summary | Avg CLIPScore: {mean_clip:.4f} | CMMD: {cmmd_val:.6f} <<<")
        cmmd_summary.append({
            "model_id": model_id,
            "model_name": model_name,
            "avg_clip_score": round(mean_clip, 6),
            "cmmd_score": round(cmmd_val, 6),
        })

    # Save CSV outputs
    detail_csv = output_root / "benchmark_30prompts_detail.csv"
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_id", "model_name", "prompt_id", "category", "prompt", "clip_score", "latency_sec", "image_path"])
        writer.writeheader()
        writer.writerows(all_results)

    summary_csv = output_root / "benchmark_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_id", "model_name", "avg_clip_score", "cmmd_score"])
        writer.writeheader()
        writer.writerows(cmmd_summary)

    print(f"\nAll 30-prompt evaluation results saved to {detail_csv} and {summary_csv}")


if __name__ == "__main__":
    main()
