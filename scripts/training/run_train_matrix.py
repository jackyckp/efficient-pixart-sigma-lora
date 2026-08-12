import argparse
import itertools
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automated training matrix runner for PixArt-Sigma LoRA."
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=1500,
        help="Maximum training steps per model in the matrix (default: 1500).",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="List of LoRA ranks to include (default: 4 8 16).",
    )
    parser.add_argument(
        "--data-scales",
        type=int,
        nargs="+",
        default=[50, 100, 260],
        help="List of dataset image counts to include (default: 50 100 260).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for matrix output checkpoints.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing runs instead of skipping them.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    root_dir = Path(__file__).resolve().parent.parent.parent
    train_script = root_dir / "scripts" / "training" / "train_local_latent_lora.py"
    latent_bundle = root_dir / "data" / "archives" / "clean_latents_512.zip"
    prompt_cache = root_dir / "data" / "features" / "t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt"

    output_root = args.output_root or (root_dir / "outputs" / "matrix_1500")

    print(
        f"Starting Training Matrix (Ranks: {args.ranks}, Data Scales: {args.data_scales}, "
        f"Steps: {args.max_train_steps}, Output: {output_root})"
    )

    total_runs = len(args.ranks) * len(args.data_scales)
    completed_count = 0

    for idx, (rank, scale) in enumerate(
        itertools.product(args.ranks, args.data_scales), start=1
    ):
        output_dir = output_root / f"r{rank}_n{scale}_steps{args.max_train_steps}"

        if not args.force and (output_dir / "run_metadata.json").is_file():
            print(f"[{idx}/{total_runs}] Skipping Rank {rank}, Scale {scale}: Already completed.")
            completed_count += 1
            continue

        print(
            f"\n[{idx}/{total_runs}] --- Running Training: Rank {rank}, Data {scale}, "
            f"Steps {args.max_train_steps} ---"
        )

        command = [
            sys.executable,
            str(train_script),
            "--latent-bundle",
            str(latent_bundle),
            "--prompt-cache",
            str(prompt_cache),
            "--num-images",
            str(scale),
            "--rank",
            str(rank),
            "--max-train-steps",
            str(args.max_train_steps),
            "--output-dir",
            str(output_dir),
            "--seed",
            "42",
        ]

        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"\nTraining failed for Rank {rank}, Data {scale}. Exiting matrix.")
            sys.exit(1)
        completed_count += 1

    print(f"\nAll {completed_count}/{total_runs} matrix training runs completed successfully!")


if __name__ == "__main__":
    main()
