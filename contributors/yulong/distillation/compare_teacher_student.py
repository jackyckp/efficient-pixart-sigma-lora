#!/usr/bin/env python3
"""Generate a controlled Teacher/Student pair and make a side-by-side sheet."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATOR = Path(__file__).resolve().parent / "generate_distilled.py"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "precomputed_prompts" / "focused_evaluation_prompts.pt"
DEFAULT_TEACHER = (
    PROJECT_ROOT
    / "models"
    / "lora_training_512"
    / "style_teacher_r16_lr1e-5_bs1_steps10000_seed42"
    / "checkpoints"
    / "step_004000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-adapter", type=Path, required=True)
    parser.add_argument("--teacher-adapter", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--teacher-steps", type=int, default=20)
    parser.add_argument("--student-steps", type=int, default=10)
    parser.add_argument("--teacher-guidance", type=float, default=2.0)
    parser.add_argument("--student-guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--memory-mode",
        choices=("cuda", "sequential_cpu_offload"),
        default="cuda",
    )
    return parser.parse_args()


def generate(
    *,
    adapter: Path,
    label: str,
    steps: int,
    guidance: float,
    output_dir: Path,
    output_file: str,
    args: argparse.Namespace,
) -> None:
    command = [
        sys.executable,
        str(GENERATOR),
        "--embeddings",
        str(args.embeddings),
        "--prompt-index",
        str(args.prompt_index),
        "--adapter",
        str(adapter),
        "--label",
        label,
        "--seed",
        str(args.seed),
        "--steps",
        str(steps),
        "--guidance",
        str(guidance),
        "--memory-mode",
        args.memory_mode,
        "--output-dir",
        str(output_dir),
        "--output-file",
        output_file,
    ]
    subprocess.run(command, check=True)


def make_sheet(
    teacher_path: Path,
    student_path: Path,
    destination: Path,
    teacher_label: str,
    student_label: str,
) -> None:
    teacher = Image.open(teacher_path).convert("RGB")
    student = Image.open(student_path).convert("RGB")
    if teacher.size != student.size:
        raise ValueError("Teacher and Student image sizes differ")
    label_height = 36
    sheet = Image.new(
        "RGB", (teacher.width + student.width, teacher.height + label_height), "white"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 11), teacher_label, fill="black", font=font)
    draw.text((teacher.width + 8, 11), student_label, fill="black", font=font)
    sheet.paste(teacher, (0, label_height))
    sheet.paste(student, (teacher.width, label_height))
    sheet.save(destination)


def main() -> int:
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (
            PROJECT_ROOT
            / "outputs"
            / "distillation"
            / f"{stamp}_p{args.prompt_index:02d}_seed{args.seed}"
        ).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_label = (
        f"Teacher step_004000 | {args.teacher_steps} DDIM | "
        f"guidance={args.teacher_guidance:g}"
    )
    student_label = (
        f"Student | {args.student_steps} DDIM | guidance={args.student_guidance:g}"
    )
    print("[1/2] Generating frozen Teacher...")
    generate(
        adapter=args.teacher_adapter,
        label="teacher_step_004000",
        steps=args.teacher_steps,
        guidance=args.teacher_guidance,
        output_dir=output_dir,
        output_file="teacher.png",
        args=args,
    )
    print("[2/2] Generating distilled Student...")
    generate(
        adapter=args.student_adapter,
        label="student",
        steps=args.student_steps,
        guidance=args.student_guidance,
        output_dir=output_dir,
        output_file="student.png",
        args=args,
    )
    sheet_path = output_dir / "teacher_vs_student.png"
    make_sheet(
        output_dir / "teacher.png",
        output_dir / "student.png",
        sheet_path,
        teacher_label,
        student_label,
    )
    comparison = {
        "teacher_adapter": str(args.teacher_adapter.resolve()),
        "student_adapter": str(args.student_adapter.resolve()),
        "embeddings": str(args.embeddings.resolve()),
        "prompt_index": args.prompt_index,
        "seed": args.seed,
        "teacher_steps": args.teacher_steps,
        "student_steps": args.student_steps,
        "teacher_guidance": args.teacher_guidance,
        "student_guidance": args.student_guidance,
        "scheduler": "DDIMScheduler",
        "timestep_spacing": "trailing",
        "sheet": str(sheet_path),
    }
    (output_dir / "comparison_metadata.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(f"Comparison saved: {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
