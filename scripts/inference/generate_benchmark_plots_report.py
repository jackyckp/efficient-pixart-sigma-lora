import argparse
import csv
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots report.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save generated plots.")
    parser.add_argument("--artifact-dir", type=Path, default=None, help="Optional artifact directory.")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[2]
    output_root = root_dir / "outputs" / "benchmark_30prompts"
    save_dir = args.output_dir or args.artifact_dir or (output_root / "plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = output_root / "benchmark_30prompts_detail.csv"
    summary_csv = output_root / "benchmark_summary.csv"

    # Read summary
    summary_rows = []
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            summary_rows.append(r)

    # Read detail
    detail_rows = []
    with open(detail_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            detail_rows.append(r)

    # Plot CMMD & CLIPScore summary bar chart
    models = [r["model_name"] for r in summary_rows]
    clip_scores = [float(r["avg_clip_score"]) for r in summary_rows]
    cmmd_scores = [float(r["cmmd_score"]) for r in summary_rows]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = "tab:blue"
    ax1.set_xlabel("Model Candidate", fontweight="bold")
    ax1.set_ylabel("Avg CLIPScore (Higher is Better)", color=color, fontweight="bold")
    bars = ax1.bar(models, clip_scores, color=color, alpha=0.6, width=0.4)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0.32, 0.38)
    plt.xticks(rotation=15, ha="right")

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("CMMD Score (Lower is Better)", color=color, fontweight="bold")
    ax2.plot(models, cmmd_scores, color=color, marker="o", linewidth=2.5, markersize=8)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("30-Prompt Benchmark: CLIPScore & CMMD Metric Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()

    chart1_path = save_dir / "cmmd_clip_benchmark.png"
    plt.savefig(chart1_path, dpi=150)
    plt.close()
    print("Saved CMMD chart to:", chart1_path)

    # Calculate category breakdown
    categories = ["Landscapes", "Flora_Fauna", "Minimalist", "Architecture"]
    cat_scores = {m: {c: [] for c in categories} for m in models}

    for r in detail_rows:
        m_name = r["model_name"]
        cat = r["category"]
        score = float(r["clip_score"])
        if m_name in cat_scores and cat in cat_scores[m_name]:
            cat_scores[m_name][cat].append(score)

    cat_means = {c: [sum(cat_scores[m][c]) / len(cat_scores[m][c]) if cat_scores[m][c] else 0.0 for m in models] for c in categories}

    # Plot Category Breakdown Bar Chart
    import numpy as np
    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    cat_colors = {"Landscapes": "#27ae60", "Flora_Fauna": "#e67e22", "Minimalist": "#2980b9", "Architecture": "#8e44ad"}
    cat_labels = {"Landscapes": "Landscapes", "Flora_Fauna": "Flora & Fauna", "Minimalist": "Minimalist", "Architecture": "Architecture"}

    for idx, (cat_key, cat_label) in enumerate(cat_labels.items()):
        offset = (idx - 1.5) * width
        ax.bar(x + offset, cat_means[cat_key], width, label=cat_label, color=cat_colors[cat_key])

    ax.set_ylabel("Avg CLIPScore", fontweight="bold")
    ax.set_title("Category-Wise CLIPScore Alignment Across 30 Validation Prompts", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0.30, 0.40)
    ax.legend(title="Prompt Category", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    chart2_path = save_dir / "category_clip_breakdown.png"
    plt.savefig(chart2_path, dpi=150)
    plt.close()
    print("Saved Category chart to:", chart2_path)

if __name__ == "__main__":
    main()
