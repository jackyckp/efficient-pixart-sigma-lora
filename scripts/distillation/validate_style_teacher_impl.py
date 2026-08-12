#!/usr/bin/env python3
"""Validate a style-teacher adapter and write normalized provenance."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Sequence

import torch

from scripts.distillation.common import (
    COMPONENT_MODEL,
    TEACHER_TIMESTEPS,
    TRANSFORMER_MODEL,
    inspect_adapter,
    repository_root,
    resolve_adapter_dir,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description="Validate and normalize a rank-16 PixArt style teacher."
    )
    parser.add_argument("--teacher-adapter", type=Path, required=True)
    parser.add_argument("--teacher-id", required=True)
    parser.add_argument("--teacher-guidance-scale", type=float, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs" / "distillation" / "teacher_manifest.json",
    )
    parser.add_argument("--transformer-model", default=TRANSFORMER_MODEL)
    parser.add_argument("--component-model", default=COMPONENT_MODEL)
    parser.add_argument(
        "--skip-fresh-load",
        action="store_true",
        help="Run structural validation only; intended for CPU tests.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def validate_teacher(args: argparse.Namespace) -> dict[str, object]:
    if not args.teacher_id.strip():
        raise ValueError("--teacher-id may not be empty.")
    if args.teacher_guidance_scale < 1.0:
        raise ValueError("--teacher-guidance-scale must be at least 1.0.")
    inspection = inspect_adapter(args.teacher_adapter)
    adapter_dir = resolve_adapter_dir(args.teacher_adapter)
    run_metadata_path = adapter_dir.parent / "run_metadata.json"
    run_metadata = None
    if run_metadata_path.is_file():
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        recorded_model = run_metadata.get("transformer_model")
        if recorded_model not in (None, args.transformer_model):
            raise ValueError(
                f"Teacher metadata base model mismatch: {recorded_model!r}."
            )

    fresh_load = {"performed": False, "status": "SKIPPED"}
    if not args.skip_fresh_load:
        from diffusers import PixArtTransformer2DModel
        from peft import PeftModel

        base = PixArtTransformer2DModel.from_pretrained(
            args.transformer_model,
            subfolder="transformer",
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=args.local_files_only,
        )
        loaded = PeftModel.from_pretrained(
            base,
            adapter_dir,
            is_trainable=False,
        ).eval()
        if loaded.peft_config["default"].r != 16:
            raise RuntimeError("Fresh-loaded teacher rank is not 16.")
        if any(parameter.requires_grad for parameter in loaded.parameters()):
            raise RuntimeError("Fresh-loaded teacher is unexpectedly trainable.")
        fresh_load = {"performed": True, "status": "PASS"}
        del loaded, base

    payload: dict[str, object] = {
        "format_version": 1,
        "status": "PASS",
        "teacher_id": args.teacher_id,
        "teacher_adapter_input": str(Path(args.teacher_adapter).resolve()),
        **inspection,
        "transformer_model": args.transformer_model,
        "component_model": args.component_model,
        "teacher_inference_steps": 20,
        "teacher_guidance_scale": args.teacher_guidance_scale,
        "teacher_timesteps": list(TEACHER_TIMESTEPS),
        "scheduler": {
            "class": "DPMSolverMultistepScheduler",
            "algorithm_type": "dpmsolver++",
            "solver_order": 2,
        },
        "source_run_metadata": (
            str(run_metadata_path) if run_metadata_path.is_file() else None
        ),
        "provenance_complete": run_metadata is not None,
        "declared_metrics_verified": False,
        "fresh_load": fresh_load,
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_teacher(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
