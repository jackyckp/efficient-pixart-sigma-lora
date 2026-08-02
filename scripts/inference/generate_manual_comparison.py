#!/usr/bin/env python3
"""Generate 3 samples from the Best Model (plant209 Step 4000) and 3 samples from the Baseline Model (Step 0) for manual comparison."""

import shutil
import subprocess
import sys
from pathlib import Path


def main():
    root_dir = Path(__file__).resolve().parents[2]
    gen_script = root_dir / "scripts" / "inference" / "generate_with_prompt.py"
    best_adapter = root_dir / "outputs" / "experiment_10k" / "r16_plant209_steps10000" / "checkpoint-4000" / "lora_adapter"
    out_dir = root_dir / "outputs" / "manual_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = Path(r"C:\Users\hklov\.gemini\antigravity-cli\brain\b70f3ed3-d14e-4144-af3b-784222775bc9")

    prompt = "A solitary pine tree standing on a misty mountain cliff, traditional Chinese ink wash painting style, shuimo hua"
    seeds = [42, 100, 2026]

    print("=== Generating 3 Samples for Best Model (plant209 Step 4000) ===")
    for idx, s in enumerate(seeds, 1):
        img_path = out_dir / f"best_model_sample_{idx}.png"
        cmd = [
            sys.executable, str(gen_script),
            "--prompt", prompt,
            "--adapter", str(best_adapter),
            "--output", str(img_path),
            "--seed", str(s),
            "--num-inference-steps", "20",
            "--guidance-scale", "1.5",
            "--allow-seen-prompt"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"Saved best model sample {idx} (seed {s}) -> {img_path}")
            shutil.copy(img_path, artifact_dir / f"best_model_sample_{idx}.png")
        else:
            print(f"Error generating best model sample {idx}: {res.stderr}")

    print("\n=== Generating 3 Samples for Baseline Model (Step 0) ===")
    for idx, s in enumerate(seeds, 1):
        img_path = out_dir / f"baseline_model_sample_{idx}.png"
        cmd = [
            sys.executable, str(gen_script),
            "--prompt", prompt,
            "--no-adapter",
            "--output", str(img_path),
            "--seed", str(s),
            "--num-inference-steps", "20",
            "--guidance-scale", "1.5",
            "--allow-seen-prompt"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"Saved baseline sample {idx} (seed {s}) -> {img_path}")
            shutil.copy(img_path, artifact_dir / f"baseline_model_sample_{idx}.png")
        else:
            print(f"Error generating baseline sample {idx}: {res.stderr}")

    print("\n✅ All 6 comparison samples generated successfully!")


if __name__ == "__main__":
    main()
