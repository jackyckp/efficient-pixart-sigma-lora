#!/usr/bin/env python3
"""Download pretrained PixArt-Sigma LoRA adapter weights (adapter_model.safetensors) from Hugging Face Hub."""

import argparse
import sys
from pathlib import Path

# Available pre-configured model repositories
PRETRAINED_MODELS = {
    "teacher_b_primary_2step": {
        "repo_id": "jackyckp/pixart-sigma-inkwash-2step-lora",
        "description": "Primary 2-step joint LoRA student (7,000 updates, 0.24s latency)",
        "subfolder": "models/teacher_b_primary_2step",
    },
    "teacher_b_primary_4step": {
        "repo_id": "jackyckp/pixart-sigma-inkwash-4step-lora",
        "description": "Primary 4-step joint LoRA student (4,500 updates, 0.48s latency)",
        "subfolder": "models/teacher_b_primary_4step",
    },
    "best_ink_wash_lora_plant209_step4000": {
        "repo_id": "jackyckp/pixart-sigma-inkwash-teacher-lora",
        "description": "20-step Style Teacher LoRA adapter (checkpoint-4000)",
        "subfolder": "models/best_ink_wash_lora_plant209_step4000",
    },
}


def download_adapter(
    repo_id: str,
    output_dir: Path,
    token: str | None = None,
    filename: str = "adapter_model.safetensors",
) -> Path:
    """Download adapter_model.safetensors and adapter_config.json from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: 'huggingface_hub' is required. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = [filename, "adapter_config.json"]
    downloaded_files = []

    print(f"Downloading from Hugging Face Hub: {repo_id}")
    for file in files_to_download:
        try:
            print(f"  -> Fetching {file}...")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=output_dir,
                token=token,
            )
            downloaded_files.append(local_path)
            print(f"  [+] Saved: {local_path}")
        except Exception as err:
            if file == "adapter_config.json":
                print(f"  [!] Note: {file} not found on hub or already local.")
            else:
                raise RuntimeError(f"Failed to download {file} from {repo_id}: {err}") from err

    safetensors_path = output_dir / filename
    if safetensors_path.is_file():
        size_mb = safetensors_path.stat().st_size / (1024 * 1024)
        print(f"✅ Successfully downloaded adapter to: {output_dir} ({size_mb:.2f} MB)")
        return output_dir
    else:
        raise FileNotFoundError(f"Download completed but {safetensors_path} was not found.")


def main():
    parser = argparse.ArgumentParser(description="Download pretrained LoRA adapter safetensors weights.")
    parser.add_argument(
        "--model",
        choices=list(PRETRAINED_MODELS.keys()) + ["all"],
        default="teacher_b_primary_2step",
        help="Which model adapter to download (default: teacher_b_primary_2step)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Custom Hugging Face repo ID (e.g. username/repo-name)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory to save adapter files (default: models/<model_name>)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face access token for private repositories",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[2]

    if args.repo_id:
        out_dir = args.output_dir if args.output_dir else (root_dir / "models" / "custom_adapter")
        download_adapter(repo_id=args.repo_id, output_dir=out_dir, token=args.token)
        return

    models_to_download = list(PRETRAINED_MODELS.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_download:
        info = PRETRAINED_MODELS[model_key]
        out_dir = args.output_dir if args.output_dir else (root_dir / info["subfolder"])
        print(f"\n=== Downloading {model_key} ===")
        print(f"Description: {info['description']}")
        try:
            download_adapter(repo_id=info["repo_id"], output_dir=out_dir, token=args.token)
        except Exception as e:
            print(f"⚠️ Could not download {model_key} from {info['repo_id']}: {e}")
            print(f"   If the Hugging Face repo is not yet published, download the weights from shared team storage.")


if __name__ == "__main__":
    main()
