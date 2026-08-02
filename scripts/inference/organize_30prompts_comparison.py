#!/usr/bin/env python3
"""Organize 30-prompt side-by-side comparison files and generate Markdown artifact report."""

import shutil
import sys
from pathlib import Path

# Add prompt definitions
VALIDATION_PROMPTS = [
    # Category 1: Landscapes (山水)
    {"id": 1, "category": "Landscapes (山水)", "prompt": "Misty mountain peaks enveloped in soft clouds, ancient pine tree on a cliff, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 2, "category": "Landscapes (山水)", "prompt": "Winding river flowing through steep mountain gorges, distant waterfall, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 3, "category": "Landscapes (山水)", "prompt": "Cascading waterfall plunging into a misty ravine, jagged rock formations, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 4, "category": "Landscapes (山水)", "prompt": "Snow-covered mountain range in winter, bare trees, frozen lake, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 5, "category": "Landscapes (山水)", "prompt": "Autumn mountains with sparse foliage, winding stone path leading to a ridge, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 6, "category": "Landscapes (山水)", "prompt": "Sunrise over sea of clouds and mountain spires, high contrast black ink brushwork, white space, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 7, "category": "Landscapes (山水)", "prompt": "Quiet lake reflecting towering mountain shadows, serene water surface, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 8, "category": "Landscapes (山水)", "prompt": "Storm clouds gathering above rugged cliffside pines, dynamic black ink splash technique, traditional Chinese ink wash painting style, shuimo hua"},

    # Category 2: Flora & Fauna (花鳥)
    {"id": 9, "category": "Flora & Fauna (花鳥)", "prompt": "Ink wash bamboo in the wind, wet brush technique, delicate leaves, subtle grey tones, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 10, "category": "Flora & Fauna (花鳥)", "prompt": "A pair of flying cranes soaring above misty clouds, elegant brushstrokes, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 11, "category": "Flora & Fauna (花鳥)", "prompt": "Blooming plum blossoms on a gnarled branch, delicate ink wash gradients, soft grey background, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 12, "category": "Flora & Fauna (花鳥)", "prompt": "Solitary eagle perched on an ancient pine branch, sharp gaze, bold black ink brushwork, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 13, "category": "Flora & Fauna (花鳥)", "prompt": "Lotus flowers blooming in a quiet pond, large wet ink leaves, dragonfly hovering, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 14, "category": "Flora & Fauna (花鳥)", "prompt": "A wild horse galloping across an open plain, dynamic ink wash style, fluid brush lines, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 15, "category": "Flora & Fauna (花鳥)", "prompt": "Wild orchids clinging to a mossy cliff, graceful curved leaves, minimalist ink wash style, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 16, "category": "Flora & Fauna (花鳥)", "prompt": "Koi fish swimming in clear water, soft ink wash ripples, transparent ink gradients, traditional Chinese ink wash painting style, shuimo hua"},

    # Category 3: Minimalist Composition (意境/留白)
    {"id": 17, "category": "Minimalist Composition (意境/留白)", "prompt": "A single small boat on a vast calm lake, minimalist composition, wide white space, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 18, "category": "Minimalist Composition (意境/留白)", "prompt": "Solitary fisherman sitting on a riverbank with a fishing rod, vast empty background, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 19, "category": "Minimalist Composition (意境/留白)", "prompt": "Single bamboo stalk in the corner of a blank paper canvas, elegant white space composition, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 20, "category": "Minimalist Composition (意境/留白)", "prompt": "A lone pine tree silhouette against a faint crescent moon, subtle grey wash, high negative space, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 21, "category": "Minimalist Composition (意境/留白)", "prompt": "Faint outline of a distant mountain peak in heavy fog, minimalist ink wash composition, wide white space, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 22, "category": "Minimalist Composition (意境/留白)", "prompt": "A single falling leaf landing on still water, delicate ink ripple lines, minimalist composition, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 23, "category": "Minimalist Composition (意境/留白)", "prompt": "Distant flight of birds vanishing into empty mist, minimalist composition, wide negative space, traditional Chinese ink wash painting style, shuimo hua"},

    # Category 4: Detailed Figures & Architecture (人物/亭台)
    {"id": 24, "category": "Architecture & Figures (人物/亭台)", "prompt": "Ancient wooden pavilion surrounded by swirling mountain fog, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 25, "category": "Architecture & Figures (人物/亭台)", "prompt": "Ancient scholar walking along a winding stone path, traditional robes, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 26, "category": "Architecture & Figures (人物/亭台)", "prompt": "Secluded stone temple tucked in a deep pine forest, mist rising, detailed architecture, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 27, "category": "Architecture & Figures (人物/亭台)", "prompt": "Traditional thatched cottage near a bamboo grove, flowing stream, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 28, "category": "Architecture & Figures (人物/亭台)", "prompt": "Ancient stone bridge spanning a misty river, small pavilion on a cliff, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 29, "category": "Architecture & Figures (人物/亭台)", "prompt": "Old scholar sitting inside a pavilion reading a book, mountain view, detailed ink wash technique, traditional Chinese ink wash painting style, shuimo hua"},
    {"id": 30, "category": "Architecture & Figures (人物/亭台)", "prompt": "Winding mountain staircase leading to a cloud-wrapped pagoda, traditional Chinese ink wash painting style, shuimo hua"},
]


def main():
    root_dir = Path(__file__).resolve().parents[2]
    src_base = root_dir / "outputs" / "benchmark_30prompts" / "generations"
    dst_dir = root_dir / "outputs" / "comparison_30prompts"

    dst_best = dst_dir / "best_model_plant209"
    dst_baseline = dst_dir / "baseline_model"
    dst_best.mkdir(parents=True, exist_ok=True)
    dst_baseline.mkdir(parents=True, exist_ok=True)

    artifact_dir = Path(r"C:\Users\hklov\.gemini\antigravity-cli\brain\b70f3ed3-d14e-4144-af3b-784222775bc9")

    print("Copying 30 prompt pairs to outputs/comparison_30prompts and artifact directory...")
    for item in VALIDATION_PROMPTS:
        p_id = item["id"]
        fname = f"prompt_{p_id:02d}.png"

        # Best Model
        src_best_file = src_base / "plant209_best" / fname
        dst_best_file = dst_best / fname
        shutil.copy(src_best_file, dst_best_file)
        shutil.copy(src_best_file, artifact_dir / f"p{p_id:02d}_best.png")

        # Baseline Model
        src_base_file = src_base / "baseline" / fname
        dst_base_file = dst_baseline / fname
        shutil.copy(src_base_file, dst_base_file)
        shutil.copy(src_base_file, artifact_dir / f"p{p_id:02d}_baseline.png")

    print("30 prompt image pairs successfully organized!")

    # Build Markdown Report
    md_content = []
    md_content.append("# 🖼️ Full 30-Prompt Side-by-Side Comparison: Best Model vs. Baseline Model\n")
    md_content.append("This document provides a side-by-side manual visual comparison for all **30 validation prompts** across 4 traditional Chinese ink wash painting categories:\n")
    md_content.append("- **Best Model**: `plant209` Step 4,000 (`outputs/experiment_10k/r16_plant209_steps10000/checkpoint-4000/lora_adapter`)\n")
    md_content.append("- **Baseline Model**: Base PixArt-Sigma (`--no-adapter`)\n")
    md_content.append("- **Full Trigger Word**: `traditional Chinese ink wash painting style, shuimo hua`\n")
    md_content.append("- **Local Comparison Folder**: [outputs/comparison_30prompts/](file:///C:/dev/efficient-pixart-sigma-lora/outputs/comparison_30prompts)\n\n")

    current_cat = ""
    for item in VALIDATION_PROMPTS:
        p_id = item["id"]
        cat = item["category"]
        prompt = item["prompt"]

        if cat != current_cat:
            current_cat = cat
            md_content.append(f"\n---\n## 🎨 Category: {current_cat}\n")

        md_content.append(f"### Prompt {p_id:02d}\n")
        md_content.append(f"> **Prompt**: *\"{prompt}\"* \n\n")
        md_content.append("| Best Model (`plant209` Step 4k) | Baseline Model (Step 0) |\n")
        md_content.append("| :---: | :---: |\n")
        md_content.append(f"| ![Best P{p_id:02d}](file:///C:/Users/hklov/.gemini/antigravity-cli/brain/b70f3ed3-d14e-4144-af3b-784222775bc9/p{p_id:02d}_best.png) | ![Baseline P{p_id:02d}](file:///C:/Users/hklov/.gemini/antigravity-cli/brain/b70f3ed3-d14e-4144-af3b-784222775bc9/p{p_id:02d}_baseline.png) |\n\n")

    report_file = artifact_dir / "full_30prompts_comparison.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("".join(md_content))

    print(f"Report artifact written to {report_file}")


if __name__ == "__main__":
    main()
