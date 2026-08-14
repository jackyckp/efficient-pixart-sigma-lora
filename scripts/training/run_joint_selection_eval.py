#!/usr/bin/env python3
"""Run Joint Candidate Selection Evaluation on plant209 checkpoints.

Calculates CLIPScore, CMMD, and Joint Selection Score:
    Selection Score = Norm(CLIPScore) - lambda * Norm(CMMD)

Rank-orders checkpoints to empirically select the winning candidate.
"""

from __future__ import annotations

import csv
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

TRIGGER_WORD = "traditional Chinese ink wash painting style, shuimo hua"

VALIDATION_PROMPTS = [
    f"Misty mountain peaks enveloped in soft clouds, ancient pine tree on a cliff, {TRIGGER_WORD}",
    f"Ink wash bamboo in the wind, wet brush technique, delicate leaves, subtle grey tones, {TRIGGER_WORD}",
    f"A single small boat on a vast calm lake, minimalist composition, wide white space, {TRIGGER_WORD}",
    f"Ancient wooden pavilion surrounded by swirling mountain fog, {TRIGGER_WORD}",
]


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


def main():
    root_dir = Path(__file__).resolve().parents[2]
    plant_dir = root_dir / "outputs" / "experiment_10k" / "r16_plant209_steps10000"
    out_csv = root_dir / "outputs" / "joint_selection_scores.csv"

    from transformers import CLIPModel, CLIPProcessor
    print("Loading CLIP Model for Joint Candidate Selection...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to("cuda").eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    ref_zip = (
        (root_dir / "data" / "ink.zip")
        if (root_dir / "data" / "ink.zip").is_file()
        else (root_dir / "data" / "archives" / "ink.zip")
    )
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
    ref_embeds = torch.cat(ref_embeds, dim=0).to("cuda")

    ckpt_dirs = sorted([d for d in plant_dir.glob("checkpoint-*") if d.is_dir()], key=lambda x: int(x.name.split("-")[1]))

    results = []
    import subprocess

    gen_script = root_dir / "scripts" / "inference" / "generate_with_prompt.py"
    eval_tmp = root_dir / "outputs" / "eval_tmp"
    generate_script = root_dir / "scripts" / "inference" / "generate_with_prompt.py"

    for ckpt in ckpt_dirs:
        step_num = int(ckpt.name.split("-")[1])
        adapter_path = ckpt / "lora_adapter"
        print(f"Evaluating plant209 Checkpoint Step {step_num}...")

        gen_embeds_list = []
        clip_scores = []

        for p_idx, prompt in enumerate(VALIDATION_PROMPTS, 1):
            img_out = eval_tmp / f"step_{step_num}_p{p_idx}.png"
            cmd = [
                sys.executable, str(generate_script),
                "--prompt", prompt,
                "--adapter", str(adapter_path),
                "--output", str(img_out),
                "--seed", "42",
                "--num-inference-steps", "20",
                "--guidance-scale", "1.5",
                "--allow-seen-prompt"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                continue

            img = Image.open(img_out).convert("RGB")
            inputs = clip_processor(text=[prompt], images=img, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = clip_model(**inputs)
                c_score = out.logits_per_image.item() / 100.0
                out_feat = clip_model.get_image_features(inputs.pixel_values)
                img_emb = getattr(out_feat, "pooler_output", out_feat) if not isinstance(out_feat, torch.Tensor) else out_feat
                gen_embeds_list.append(img_emb.cpu())
            clip_scores.append(c_score)

        gen_embeds = torch.cat(gen_embeds_list, dim=0).to("cuda")
        cmmd_val = compute_cmmd(gen_embeds, ref_embeds)
        mean_clip = sum(clip_scores) / len(clip_scores)

        results.append({
            "step": step_num,
            "clip_score": mean_clip,
            "cmmd": cmmd_val,
        })
        print(f"  Step {step_num:5d} | CLIPScore: {mean_clip:.4f} | CMMD: {cmmd_val:.6f}")

    # Compute Normalized Joint Score
    min_clip = min(r["clip_score"] for r in results)
    max_clip = max(r["clip_score"] for r in results)
    clip_range = max_clip - min_clip if max_clip > min_clip else 1.0

    min_cmmd = min(r["cmmd"] for r in results)
    max_cmmd = max(r["cmmd"] for r in results)
    cmmd_range = max_cmmd - min_cmmd if max_cmmd > min_cmmd else 1.0

    lambda_cmmd = 1.0
    for r in results:
        norm_clip = (r["clip_score"] - min_clip) / clip_range
        norm_cmmd = (r["cmmd"] - min_cmmd) / cmmd_range
        r["norm_clip"] = round(norm_clip, 4)
        r["norm_cmmd"] = round(norm_cmmd, 4)
        r["joint_score"] = round(norm_clip - lambda_cmmd * norm_cmmd, 4)

    results.sort(key=lambda x: x["joint_score"], reverse=True)

    print("\n=== JOINT SELECTION SCORE RANKINGS ===")
    print(f"{'Rank':<5} | {'Step':<6} | {'CLIPScore':<10} | {'CMMD':<10} | {'Norm(CLIP)':<10} | {'Norm(CMMD)':<10} | {'Joint Score':<11}")
    print("-" * 75)
    for rank, r in enumerate(results, 1):
        print(f"{rank:<5} | {r['step']:<6} | {r['clip_score']:<10.4f} | {r['cmmd']:<10.6f} | {r['norm_clip']:<10.4f} | {r['norm_cmmd']:<10.4f} | {r['joint_score']:<11.4f}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "clip_score", "cmmd", "norm_clip", "norm_cmmd", "joint_score"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved joint selection scores to {out_csv}")


if __name__ == "__main__":
    main()
