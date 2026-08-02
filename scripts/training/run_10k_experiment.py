#!/usr/bin/env python3
"""Run 10k-step LoRA training matrix for Rank 16 across 4 datasets (50, 100, 209 plant, 260 full).

Saves intermediate checkpoints every 1,000 steps and computes CLIPScore & latency metrics at guidance_scale=1.5.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="10k-step LoRA training experiment runner (Rank 16)."
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=10000,
        help="Total training steps per run (default: 10000).",
    )
    parser.add_argument(
        "--checkpointing-steps",
        type=int,
        default=1000,
        help="Checkpoint frequency (default: 1000).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.5,
        help="Guidance scale for evaluation inference (default: 1.5).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for 10k experiment outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser


def run_training_configs(args: argparse.Namespace, root_dir: Path, output_root: Path) -> list[dict[str, object]]:
    train_script = root_dir / "scripts" / "training" / "train_local_latent_lora.py"
    latent_bundle = root_dir / "data" / "archives" / "clean_latents_512.zip"
    prompt_cache = root_dir / "data" / "features" / "t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt"

    configs = [
        {"dataset": "n50", "num_images": 50, "plant_only": False},
        {"dataset": "n100", "num_images": 100, "plant_only": False},
        {"dataset": "plant209", "num_images": 209, "plant_only": True},
        {"dataset": "n260", "num_images": 260, "plant_only": False},
    ]

    executed_runs = []

    for idx, cfg in enumerate(configs, start=1):
        ds_name = cfg["dataset"]
        num_images = cfg["num_images"]
        is_plant = cfg["plant_only"]
        out_dir = output_root / f"r16_{ds_name}_steps{args.max_train_steps}"

        print(f"\n[{idx}/{len(configs)}] === Training Rank 16, Dataset '{ds_name}' ({num_images} images), Steps {args.max_train_steps} ===")

        cmd = [
            sys.executable,
            str(train_script),
            "--latent-bundle", str(latent_bundle),
            "--prompt-cache", str(prompt_cache),
            "--num-images", str(num_images),
            "--rank", "16",
            "--max-train-steps", str(args.max_train_steps),
            "--checkpointing-steps", str(args.checkpointing_steps),
            "--output-dir", str(out_dir),
            "--seed", str(args.seed),
        ]
        if is_plant:
            cmd.append("--plant-only")

        if (out_dir / "run_metadata.json").is_file():
            print(f"Skipping training for {ds_name}: Already completed.")
        else:
            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"Training failed for dataset {ds_name}.")
                sys.exit(1)

        executed_runs.append({
            "dataset_name": ds_name,
            "num_images": num_images,
            "plant_only": is_plant,
            "output_dir": out_dir,
        })

    return executed_runs


def evaluate_checkpoints(
    runs: list[dict[str, object]],
    args: argparse.Namespace,
    root_dir: Path,
    output_root: Path,
):
    import matplotlib.pyplot as plt
    from transformers import CLIPModel, CLIPProcessor

    print("\n=== Initializing CLIP Model for Metrics Evaluation ===")
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True).to("cuda").eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    generate_script = root_dir / "scripts" / "inference" / "generate_with_prompt.py"
    eval_prompt = "A solitary pine tree standing on a misty mountain cliff, traditional Chinese ink wash painting style, shuimo hua"

    metrics_rows = []

    # 1. Evaluate Baseline Model (Step 0)
    print("\n--- Evaluating Baseline Model (Un-adapted Base PixArt-Sigma) ---")
    base_img_out = output_root / "eval_generations" / "baseline" / "step_0.png"
    base_img_out.parent.mkdir(parents=True, exist_ok=True)
    base_cmd = [
        sys.executable,
        str(generate_script),
        "--prompt", eval_prompt,
        "--no-adapter",
        "--output", str(base_img_out),
        "--seed", str(args.seed),
        "--num-inference-steps", "20",
        "--guidance-scale", str(args.guidance_scale),
    ]
    start_t = time.perf_counter()
    res = subprocess.run(base_cmd, capture_output=True, text=True)
    base_latency = time.perf_counter() - start_t
    if res.returncode == 0:
        base_img = Image.open(base_img_out).convert("RGB")
        inputs = clip_processor(text=[eval_prompt], images=base_img, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            outputs = clip_model(**inputs)
            base_clip_score = (outputs.logits_per_image.item() / 100.0)
        print(f"Step     0 (Baseline Model) | CLIPScore: {base_clip_score:.4f} | Latency: {base_latency:.2f}s | Saved: {base_img_out.name}")
        metrics_rows.append({
            "dataset": "baseline",
            "num_images": 0,
            "rank": 0,
            "step": 0,
            "guidance_scale": args.guidance_scale,
            "clip_score": round(base_clip_score, 6),
            "latency_sec": round(base_latency, 4),
            "image_path": str(base_img_out),
        })

    # 2. Evaluate Checkpoints for Each Fine-Tuned Model
    for run in runs:
        out_dir: Path = run["output_dir"]
        ds_name: str = run["dataset_name"]

        step_checkpoints = sorted(
            [d for d in out_dir.glob("checkpoint-*") if d.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if (out_dir / "lora_adapter").is_dir():
            step_checkpoints.append(out_dir)

        print(f"\n--- Evaluating Checkpoints for Dataset '{ds_name}' ({len(step_checkpoints)} checkpoints) ---")

        for ckpt in step_checkpoints:
            step_num = int(ckpt.name.split("-")[-1]) if ckpt.name.startswith("checkpoint-") else args.max_train_steps
            adapter_path = ckpt / "lora_adapter" if (ckpt / "lora_adapter").is_dir() else ckpt

            img_out = output_root / "eval_generations" / ds_name / f"step_{step_num}.png"
            img_out.parent.mkdir(parents=True, exist_ok=True)

            gen_cmd = [
                sys.executable,
                str(generate_script),
                "--prompt", eval_prompt,
                "--adapter", str(adapter_path),
                "--output", str(img_out),
                "--seed", str(args.seed),
                "--num-inference-steps", "20",
                "--guidance-scale", str(args.guidance_scale),
                "--allow-seen-prompt",
            ]

            start_t = time.perf_counter()
            res = subprocess.run(gen_cmd, capture_output=True, text=True)
            latency_sec = time.perf_counter() - start_t

            if res.returncode != 0:
                print(f"Generation failed for {ckpt.name}: {res.stderr}")
                continue

            # Calculate CLIP score
            image = Image.open(img_out).convert("RGB")
            inputs = clip_processor(text=[eval_prompt], images=image, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                clip_score = (logits_per_image.item() / 100.0)

            print(f"Step {step_num:5d} | CLIPScore: {clip_score:.4f} | Latency: {latency_sec:.2f}s | Saved: {img_out.name}")

            metrics_rows.append({
                "dataset": ds_name,
                "num_images": run["num_images"],
                "rank": 16,
                "step": step_num,
                "guidance_scale": args.guidance_scale,
                "clip_score": round(clip_score, 6),
                "latency_sec": round(latency_sec, 4),
                "image_path": str(img_out),
            })

    metrics_csv = output_root / "metrics_10k.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "num_images", "rank", "step", "guidance_scale", "clip_score", "latency_sec", "image_path"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    print(f"\n✅ All 10k experiment metrics successfully exported to {metrics_csv}")

    # Generate Visualization Plots
    try:
        plot_trajectory_chart(metrics_rows, output_root)
    except Exception as err:
        print(f"Plotting chart warning: {err}")

    return metrics_rows


def plot_trajectory_chart(metrics_rows: list[dict[str, object]], output_root: Path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 6))
    colors = {"baseline": "#7f8c8d", "n50": "#e74c3c", "n100": "#e67e22", "plant209": "#2ecc71", "n260": "#3498db"}
    styles = {"baseline": "--", "n50": "-", "n100": "-", "plant209": "-", "n260": "-"}

    grouped = {}
    for row in metrics_rows:
        ds = str(row["dataset"])
        grouped.setdefault(ds, []).append(row)

    base_score = None
    if "baseline" in grouped and grouped["baseline"]:
        base_score = grouped["baseline"][0]["clip_score"]
        plt.axhline(y=base_score, color=colors["baseline"], linestyle="--", linewidth=2.0, label=f"Baseline (Step 0): {base_score:.4f}")

    for ds, rows in grouped.items():
        if ds == "baseline":
            continue
        sorted_rows = sorted(rows, key=lambda r: int(r["step"]))
        steps = [int(r["step"]) for r in sorted_rows]
        scores = [float(r["clip_score"]) for r in sorted_rows]

        # Add step 0 baseline point to each series
        if base_score is not None:
            steps = [0] + steps
            scores = [base_score] + scores

        plt.plot(steps, scores, marker="o", linewidth=2.2, color=colors.get(ds, "purple"), label=f"Dataset: {ds}")

    plt.title("10,000-Step CLIPScore Trajectory vs. Baseline (Rank 16, Guidance=1.5)", fontsize=13, fontweight="bold")
    plt.xlabel("Training Steps (0 to 10,000)", fontsize=11)
    plt.ylabel("CLIPScore (Text-Image Alignment)", fontsize=11)
    plt.xticks(range(0, 11000, 1000))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Dataset Configuration", fontsize=10)
    plt.tight_layout()

    chart_path = output_root / "clip_trajectory_10k.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved to {chart_path}")


def main():
    args = build_parser().parse_args()
    root_dir = repository_root()
    output_root = args.output_root or (root_dir / "outputs" / "experiment_10k")
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Starting 10k Experiment (Rank 16, Steps: {args.max_train_steps}, Checkpoint Freq: {args.checkpointing_steps}, Guidance: {args.guidance_scale})")
    runs = run_training_configs(args, root_dir, output_root)
    evaluate_checkpoints(runs, args, root_dir, output_root)


if __name__ == "__main__":
    main()
