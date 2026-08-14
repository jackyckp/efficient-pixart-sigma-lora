#!/usr/bin/env python3
"""Generate a student-only evaluation set against validated teacher images."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.distillation.common import (  # noqa: E402
    COMPONENT_MODEL,
    TRANSFORMER_MODEL,
    resolve_adapter_dir,
    write_json,
)
from scripts.distillation.generate_evaluation_set_impl import (  # noqa: E402
    _load_records,
    _student_latent,
)
from scripts.distillation.evaluation_prompt_cache import (  # noqa: E402
    DEFAULT_CACHE,
    load_evaluation_prompt_cache,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate only distilled student images and reuse teacher references."
    )
    parser.add_argument("--student-adapter", type=Path, required=True)
    parser.add_argument("--student-steps", type=int, choices=(2, 4), required=True)
    parser.add_argument("--evaluation-prompts", type=Path, required=True)
    parser.add_argument(
        "--evaluation-prompt-cache", type=Path, default=DEFAULT_CACHE
    )
    parser.add_argument("--reference-teacher-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-limit", type=int, default=30)
    parser.add_argument("--seeds-per-prompt", type=int, default=4)
    parser.add_argument("--transformer-model", default=TRANSFORMER_MODEL)
    parser.add_argument("--component-model", default=COMPONENT_MODEL)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _encode_unique_prompts(
    records: Sequence[dict[str, Any]], args: argparse.Namespace
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return load_evaluation_prompt_cache(
        args.evaluation_prompt_cache,
        records,
        component_model=args.component_model,
    )


def validate_teacher_references(
    directory: Path,
    records: Sequence[dict[str, Any]],
) -> None:
    for record in records:
        image_path = directory / record["filename"]
        metadata_path = image_path.with_suffix(".json")
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing teacher reference: {image_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing teacher metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "status": "PASS",
            "prompt_id": record["prompt_id"],
            "prompt": record["prompt"],
            "seed": record["seed"],
            "num_inference_steps": 20,
            "transformer_forward_calls": 20,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Teacher reference metadata mismatch: {metadata_path}")


def generate_student_set(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")
    if args.prompt_limit <= 0 or not 1 <= args.seeds_per_prompt <= 4:
        raise ValueError("Invalid prompt/seed limits.")
    args.evaluation_prompts = args.evaluation_prompts.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir = args.reference_teacher_images.resolve()
    records = _load_records(args)
    validate_teacher_references(teacher_dir, records)
    prompt_features = _encode_unique_prompts(records, args)

    from diffusers import AutoencoderKL, DDPMScheduler, PixArtTransformer2DModel
    from diffusers.image_processor import VaeImageProcessor
    from peft import PeftModel

    device = torch.device("cuda")
    base = PixArtTransformer2DModel.from_pretrained(
        args.transformer_model,
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    )
    student = PeftModel.from_pretrained(
        base,
        resolve_adapter_dir(args.student_adapter),
        is_trainable=False,
    ).to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(
        args.component_model,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )
    alphas = scheduler.alphas_cumprod.float()
    results: list[tuple[dict[str, Any], torch.Tensor, float, int]] = []
    for record in records:
        embed, mask = prompt_features[record["prompt_id"]]
        latent, seconds, calls = _student_latent(
            student,
            alphas,
            embed.to(device),
            mask.to(device),
            record["seed"],
            args.student_steps,
            device,
        )
        if calls != args.student_steps or not bool(torch.isfinite(latent).all()):
            raise RuntimeError("Student exact-call or finite-latent contract failed.")
        results.append((record, latent, seconds, calls))
    del student, base
    gc.collect()
    torch.cuda.empty_cache()

    vae = AutoencoderKL.from_pretrained(
        args.component_model,
        subfolder="vae",
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    processor = VaeImageProcessor(vae_scale_factor=8)
    student_dir = args.output_dir / "student"
    student_dir.mkdir(parents=True, exist_ok=True)
    for record, latent, seconds, calls in results:
        path = student_dir / record["filename"]
        metadata_path = path.with_suffix(".json")
        if path.is_file() and metadata_path.is_file() and not args.overwrite:
            continue
        with torch.inference_mode():
            decoded = vae.decode(
                latent.to(device) / vae.config.scaling_factor,
                return_dict=False,
            )[0]
        image = processor.postprocess(decoded, output_type="pil")[0]
        image.save(path)
        write_json(
            metadata_path,
            {
                "status": "PASS",
                **record,
                "num_inference_steps": args.student_steps,
                "guidance_scale": 1.0,
                "classifier_free_guidance_branch": False,
                "transformer_forward_calls": calls,
                "denoise_seconds": seconds,
            },
        )
    result = {
        "status": "PASS",
        "mode": "student_only_with_validated_teacher_reuse",
        "student_adapter": str(resolve_adapter_dir(args.student_adapter)),
        "student_steps": args.student_steps,
        "prompt_count": args.prompt_limit,
        "seeds_per_prompt": args.seeds_per_prompt,
        "image_count_per_model": len(records),
        "teacher_images": str(teacher_dir),
        "student_images": str(student_dir),
    }
    write_json(args.output_dir / "generation_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    generate_student_set(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

