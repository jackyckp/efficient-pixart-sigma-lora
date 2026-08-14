#!/usr/bin/env python3
"""Generate 30 validation prompt images for Step 7000 and Step 10000 plant209 models and update full_30prompts_comparison.md with a 4-column side-by-side matrix."""

import shutil
import subprocess
import sys
import time
from pathlib import Path

TRIGGER_WORD = "traditional Chinese ink wash painting style, shuimo hua"

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
    {"id": 9, "category": "Flora & Fauna", "prompt": f"Ink wash bamboo in the wind, wet brush technique, delicate leaves, subtle grey tones, {TRIGGER_WORD}"},
    {"id": 10, "category": "Flora & Fauna", "prompt": f"A pair of flying cranes soaring above misty clouds, elegant brushstrokes, {TRIGGER_WORD}"},
    {"id": 11, "category": "Flora & Fauna", "prompt": f"Blooming plum blossoms on a gnarled branch, delicate ink wash gradients, soft grey background, {TRIGGER_WORD}"},
    {"id": 12, "category": "Flora & Fauna", "prompt": f"Solitary eagle perched on an ancient pine branch, sharp gaze, bold black ink brushwork, {TRIGGER_WORD}"},
    {"id": 13, "category": "Flora & Fauna", "prompt": f"Lotus flowers blooming in a quiet pond, large wet ink leaves, dragonfly hovering, {TRIGGER_WORD}"},
    {"id": 14, "category": "Flora & Fauna", "prompt": f"A wild horse galloping across an open plain, dynamic ink wash style, fluid brush lines, {TRIGGER_WORD}"},
    {"id": 15, "category": "Flora & Fauna", "prompt": f"Wild orchids clinging to a mossy cliff, graceful curved leaves, minimalist ink wash style, {TRIGGER_WORD}"},
    {"id": 16, "category": "Flora & Fauna", "prompt": f"Koi fish swimming in clear water, soft ink wash ripples, transparent ink gradients, {TRIGGER_WORD}"},

    # Category 3: Minimalist Composition 
    {"id": 17, "category": "Minimalist Composition", "prompt": f"A single small boat on a vast calm lake, minimalist composition, wide white space, {TRIGGER_WORD}"},
    {"id": 18, "category": "Minimalist Composition", "prompt": f"Solitary fisherman sitting on a riverbank with a fishing rod, vast empty background, {TRIGGER_WORD}"},
    {"id": 19, "category": "Minimalist Composition", "prompt": f"Single bamboo stalk in the corner of a blank paper canvas, elegant white space composition, {TRIGGER_WORD}"},
    {"id": 20, "category": "Minimalist Composition", "prompt": f"A lone pine tree silhouette against a faint crescent moon, subtle grey wash, high negative space, {TRIGGER_WORD}"},
    {"id": 21, "category": "Minimalist Composition", "prompt": f"Faint outline of a distant mountain peak in heavy fog, minimalist ink wash composition, wide white space, {TRIGGER_WORD}"},
    {"id": 22, "category": "Minimalist Composition", "prompt": f"A single falling leaf landing on still water, delicate ink ripple lines, minimalist composition, {TRIGGER_WORD}"},
    {"id": 23, "category": "Minimalist Composition", "prompt": f"Distant flight of birds vanishing into empty mist, minimalist composition, wide negative space, {TRIGGER_WORD}"},

    # Category 4: Detailed Figures & Architecture 
    {"id": 24, "category": "Architecture & Figures", "prompt": f"Ancient wooden pavilion surrounded by swirling mountain fog, {TRIGGER_WORD}"},
    {"id": 25, "category": "Architecture & Figures", "prompt": f"Ancient scholar walking along a winding stone path, traditional robes, {TRIGGER_WORD}"},
    {"id": 26, "category": "Architecture & Figures", "prompt": f"Secluded stone temple tucked in a deep pine forest, mist rising, detailed architecture, {TRIGGER_WORD}"},
    {"id": 27, "category": "Architecture & Figures", "prompt": f"Traditional thatched cottage near a bamboo grove, flowing stream, {TRIGGER_WORD}"},
    {"id": 28, "category": "Architecture & Figures", "prompt": f"Ancient stone bridge spanning a misty river, small pavilion on a cliff, {TRIGGER_WORD}"},
    {"id": 29, "category": "Architecture & Figures", "prompt": f"Old scholar sitting inside a pavilion reading a book, mountain view, detailed ink wash technique, {TRIGGER_WORD}"},
    {"id": 30, "category": "Architecture & Figures", "prompt": f"Winding mountain staircase leading to a cloud-wrapped pagoda, {TRIGGER_WORD}"},
]


def main():
    root_dir = Path(__file__).resolve().parents[2]
    gen_script = root_dir / "scripts" / "inference" / "generate_with_prompt.py"

    plant_dir = root_dir / "outputs" / "experiment_10k" / "r16_plant209_steps10000"
    adapter_7000 = plant_dir / "checkpoint-7000" / "lora_adapter"
    adapter_10000 = plant_dir / "checkpoint-10000" / "lora_adapter"

    dst_dir = root_dir / "outputs" / "comparison_30prompts"
    dst_7000 = dst_dir / "plant209_step7000"
    dst_10000 = dst_dir / "plant209_step10000"
    dst_7000.mkdir(parents=True, exist_ok=True)
    dst_10000.mkdir(parents=True, exist_ok=True)

    reports_dir = root_dir / "outputs" / "reports"
    images_dir = reports_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    models_to_gen = [
        ("step7000", "Step 7,000", adapter_7000, dst_7000),
        ("step10000", "Step 10,000", adapter_10000, dst_10000),
    ]

    generate_script = root_dir / "scripts" / "inference" / "generate_with_prompt.py"

    for m_key, m_name, adapter_path, out_subfolder in models_to_gen:
        print(f"\n=======================================================")
        print(f"  Generating 30 Prompts for Model: {m_name}")
        print(f"=======================================================")

        for item in VALIDATION_PROMPTS:
            p_id = item["id"]
            cat = item["category"]
            prompt = item["prompt"]

            img_out = out_subfolder / f"prompt_{p_id:02d}.png"
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
            t0 = time.perf_counter()
            res = subprocess.run(cmd, capture_output=True, text=True)
            latency = time.perf_counter() - t0

            if res.returncode == 0:
                print(f"  [{cat[:12]:12s}] P{p_id:02d} | Latency: {latency:.2f}s -> {img_out.name}")
                shutil.copy(img_out, images_dir / f"p{p_id:02d}_{m_key}.png")
            else:
                print(f"Error prompt {p_id}: {res.stderr}")

    print("\nBuilding updated 4-column 30-prompt Markdown comparison report...")

    md_content = []
    md_content.append("# 🖼️ Full 30-Prompt Multi-Step Comparison: Baseline vs Step 4k vs Step 7k vs Step 10k\n\n")
    md_content.append("This document presents a 4-column side-by-side visual comparison for all **30 validation prompts** across 4 traditional Chinese ink wash painting categories:\n\n")
    md_content.append("1. **Baseline Model (Step 0)**: Base PixArt-Sigma (`--no-adapter`)\n")
    md_content.append("2. **Step 4,000 Model**: Rank 4 candidate (`plant209` Step 4,000 / Optimal Ink Bleed & Diffusion)\n")
    md_content.append("3. **Step 7,000 Model**: Rank 1 candidate (`plant209` Step 7,000 / Highest Joint Selection Score `+0.7970`)\n")
    md_content.append("4. **Step 10,000 Model**: Rank 6 candidate (`plant209` Step 10,000 / High Step Saturation)\n\n")
    md_content.append("- **Sampling**: `guidance_scale = 1.5`, `seed = 42`, `num_inference_steps = 20`\n")
    md_content.append("- **Local Comparison Folder**: `outputs/comparison_30prompts/`\n\n")

    current_cat = ""
    for item in VALIDATION_PROMPTS:
        p_id = item["id"]
        cat = item["category"]
        prompt = item["prompt"]

        if cat != current_cat:
            current_cat = cat
            md_content.append(f"\n---\n## 🎨 Category: {current_cat}\n\n")

        md_content.append(f"### Prompt {p_id:02d}\n\n")
        md_content.append(f"> **Prompt**: *\"{prompt}\"*\n\n")
        md_content.append("| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |\n")
        md_content.append("| :---: | :---: | :---: | :---: |\n")
        md_content.append(
            f"| ![Baseline P{p_id:02d}](./images/p{p_id:02d}_baseline.png) "
            f"| ![Step 4k P{p_id:02d}](./images/p{p_id:02d}_best.png) "
            f"| ![Step 7k P{p_id:02d}](./images/p{p_id:02d}_step7000.png) "
            f"| ![Step 10k P{p_id:02d}](./images/p{p_id:02d}_step10000.png) |\n\n"
        )

    report_path = reports_dir / "full_30prompts_comparison.md"
    report_path.write_text("".join(md_content), encoding="utf-8")
    print(f"\n✅ Updated 4-column 30-prompt comparison report written to {report_path}")


if __name__ == "__main__":
    main()
