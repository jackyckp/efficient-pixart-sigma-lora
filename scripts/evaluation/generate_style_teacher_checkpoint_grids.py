#!/usr/bin/env python3
"""Generate labelled 2x2 grids for the style-teacher checkpoint sweep."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont

TRANSFORMER_MODEL = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
COMPONENT_MODEL = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
MAX_SEQUENCE_LENGTH = 300
DEFAULT_RANKS = (4, 8, 16, 32)
DEFAULT_PROMPT = (
    "A solitary white crane gliding above a misty lotus pond at dawn, "
    "distant mountains fading into pale ink, sparse composition, "
    "Chinese ink wash painting style, sumi-e"
)
STEP_PATTERN = re.compile(r"step_(\d{6})$")


@dataclass(frozen=True)
class Checkpoint:
    rank: int
    step: int
    adapter_dir: Path
    metadata: dict[str, Any]


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Generate same-prompt, same-seed grids for teacher checkpoints."
    )
    value.add_argument(
        "--model-root",
        type=Path,
        default=root() / "outputs" / "style_teacher" / "all_n260_steps10000",
    )
    value.add_argument(
        "--output-dir",
        type=Path,
        default=root() / "outputs" / "evaluation" / "style_teacher_checkpoint_grids",
    )
    value.add_argument("--prompt", default=DEFAULT_PROMPT)
    value.add_argument("--negative-prompt", default="")
    value.add_argument("--seed", type=int, default=123)
    value.add_argument("--num-inference-steps", type=int, default=20)
    value.add_argument("--guidance-scale", type=float, default=1.5)
    value.add_argument("--expected-groups", type=int, default=10)
    value.add_argument("--ranks", type=int, nargs="+", default=DEFAULT_RANKS)
    value.add_argument("--t5-gpu-memory", default="4GiB")
    value.add_argument("--t5-cpu-memory", default="8GiB")
    value.add_argument("--transformer-model", default=TRANSFORMER_MODEL)
    value.add_argument("--component-model", default=COMPONENT_MODEL)
    value.add_argument("--allow-seen-prompt", action="store_true")
    value.add_argument("--include-base-model", action="store_true")
    value.add_argument("--prompt-source-sample-id", default=None)
    value.add_argument("--dry-run", action="store_true")
    return value


def run_directory(model_root: Path, rank: int) -> Path:
    return model_root / f"r{rank}_lr1e-05"


def read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON: {path}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def discover_checkpoint_groups(
    model_root: Path, ranks: Sequence[int], expected_groups: int
) -> dict[int, list[Checkpoint]]:
    per_rank: dict[int, dict[int, Checkpoint]] = {}
    for rank in ranks:
        directory = run_directory(model_root, rank)
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing rank-{rank} directory: {directory}")
        checkpoints: dict[int, Checkpoint] = {}
        for checkpoint_dir in (directory / "checkpoints").glob("step_*"):
            matched = STEP_PATTERN.fullmatch(checkpoint_dir.name)
            if not matched:
                continue
            adapter = checkpoint_dir / "lora_adapter"
            if not (adapter / "adapter_config.json").is_file():
                raise FileNotFoundError(f"Missing adapter: {adapter}")
            step = int(matched.group(1))
            if step in checkpoints:
                raise ValueError(f"Duplicate rank-{rank} checkpoint: {step}")
            metadata_file = checkpoint_dir / "checkpoint_metadata.json"
            checkpoints[step] = Checkpoint(
                rank, step, adapter, read_json(metadata_file) if metadata_file.is_file() else {}
            )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found: {directory}")
        per_rank[rank] = checkpoints
    common = set.intersection(*(set(value) for value in per_rank.values()))
    union = set.union(*(set(value) for value in per_rank.values()))
    if common != union:
        missing = {
            str(rank): sorted(union - set(value))
            for rank, value in per_rank.items()
            if set(value) != union
        }
        raise ValueError(f"Ranks do not have identical checkpoint steps: {missing}")
    if len(common) != expected_groups:
        raise ValueError(
            f"Expected {expected_groups} checkpoint groups, found {len(common)}: {sorted(common)}"
        )
    return {step: [per_rank[rank][step] for rank in ranks] for step in sorted(common)}


def audit_prompt(
    prompt: str, model_root: Path, ranks: Sequence[int], allow_seen: bool
) -> dict[str, Any]:
    normalized = " ".join(prompt.split()).casefold()
    seen_ids: set[str] = set()
    matches: list[dict[str, str]] = []
    manifests: list[str] = []
    for rank in ranks:
        manifest = run_directory(model_root, rank) / "subset_manifest.json"
        if not manifest.is_file():
            continue
        manifests.append(str(manifest))
        try:
            rows = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read manifest: {manifest}") from exc
        if not isinstance(rows, list):
            raise ValueError(f"Invalid manifest: {manifest}")
        for row in rows:
            sample_id, caption = row.get("sample_id"), row.get("caption")
            if not isinstance(sample_id, str) or not isinstance(caption, str):
                raise ValueError(f"Invalid row in manifest: {manifest}")
            if sample_id in seen_ids:
                continue
            seen_ids.add(sample_id)
            if " ".join(caption.split()).casefold() == normalized:
                matches.append({"sample_id": sample_id, "caption": caption})
    result: dict[str, Any] = {
        "training_manifests": manifests,
        "unique_training_sample_ids_checked": len(seen_ids),
        "exact_training_caption_match": bool(matches),
        "matches": matches,
    }
    if matches and not allow_seen:
        raise ValueError("Prompt exactly matches a training caption; use --allow-seen-prompt.")
    return result


def encode_prompts(
    args: argparse.Namespace, offload: Path
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, float]:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(args.component_model, subfolder="tokenizer")
    offload.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    encoder = T5EncoderModel.from_pretrained(
        args.component_model,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: args.t5_gpu_memory, "cpu": args.t5_cpu_memory},
        offload_folder=str(offload),
        offload_state_dict=True,
        low_cpu_mem_usage=True,
    ).eval()
    texts = [args.prompt] + ([args.negative_prompt] if args.guidance_scale > 1 else [])
    tokens = tokenizer(
        texts,
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    device = encoder.get_input_embeddings().weight.device
    with torch.inference_mode():
        embeddings = encoder(
            input_ids=tokens.input_ids.to(device),
            attention_mask=tokens.attention_mask.to(device),
        ).last_hidden_state.to("cpu", dtype=torch.float16)
    masks = tokens.attention_mask.to("cpu")
    seconds = time.perf_counter() - started
    positive, positive_mask = embeddings[:1].contiguous(), masks[:1].contiguous()
    negative = negative_mask = None
    if args.guidance_scale > 1:
        negative, negative_mask = embeddings[1:2].contiguous(), masks[1:2].contiguous()
    del embeddings, masks, tokens, encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return positive, positive_mask, negative, negative_mask, seconds


def get_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    ):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    text_font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = box
    draw.text(
        ((left + right) // 2, (top + bottom) // 2),
        text,
        font=text_font,
        fill="black",
        anchor="mm",
    )


def checkpoint_label(checkpoint: Checkpoint) -> str:
    loss = checkpoint.metadata.get("loss")
    if isinstance(loss, (int, float)) and math.isfinite(loss):
        return f"Rank {checkpoint.rank}  |  loss {loss:.5f}"
    return f"Rank {checkpoint.rank}  |  lr 1e-5"


def build_grid(
    images: Sequence[Image.Image],
    checkpoints: Sequence[Checkpoint],
    step: int,
    args: argparse.Namespace,
    output: Path,
) -> None:
    if len(images) != 4 or len(checkpoints) != 4:
        raise ValueError("A checkpoint grid requires four images.")
    tile, title, label = 512, 58, 42
    canvas = Image.new("RGB", (tile * 2, title + (tile + label) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    draw_centered(
        draw,
        (0, 0, canvas.width, title),
        (
            f"Style teacher checkpoint {step:,}  |  "
            f"{args.num_inference_steps} inference steps  |  "
            f"CFG {args.guidance_scale:g}  |  seed {args.seed}"
        ),
        get_font(23),
    )
    for index, (image, checkpoint) in enumerate(zip(images, checkpoints)):
        column, row = index % 2, index // 2
        x, y = column * tile, title + row * (tile + label)
        canvas.paste(image.convert("RGB").resize((tile, tile)), (x, y))
        draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline="#666666", width=2)
        draw_centered(
            draw,
            (x, y + tile, x + tile, y + tile + label),
            checkpoint_label(checkpoint),
            get_font(22),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def generate_image(
    checkpoint: Checkpoint,
    args: argparse.Namespace,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    negative_embeds: torch.Tensor | None,
    negative_mask: torch.Tensor | None,
) -> tuple[Image.Image, float]:
    from diffusers import PixArtSigmaPipeline, PixArtTransformer2DModel
    from peft import PeftModel

    base = PixArtTransformer2DModel.from_pretrained(
        args.transformer_model,
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    loaded = PeftModel.from_pretrained(base, checkpoint.adapter_dir, is_trainable=False).eval()
    if loaded.peft_config["default"].r != checkpoint.rank:
        raise ValueError(f"Adapter rank mismatch: {checkpoint.adapter_dir}")
    pipe = PixArtSigmaPipeline.from_pretrained(
        args.component_model,
        transformer=loaded.get_base_model().eval(),
        text_encoder=None,
        tokenizer=None,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    kwargs: dict[str, Any] = {
        "prompt": None,
        "negative_prompt": None,
        "prompt_embeds": prompt_embeds.to("cuda"),
        "prompt_attention_mask": prompt_mask.to("cuda"),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": 512,
        "width": 512,
        "use_resolution_binning": False,
        "generator": torch.Generator("cuda").manual_seed(args.seed),
    }
    if negative_embeds is not None and negative_mask is not None:
        kwargs.update(
            negative_prompt_embeds=negative_embeds.to("cuda"),
            negative_prompt_attention_mask=negative_mask.to("cuda"),
        )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        image = pipe(**kwargs).images[0]
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    del pipe, loaded, base
    gc.collect()
    torch.cuda.empty_cache()
    return image, seconds


def generate_official_base_image(
    args: argparse.Namespace,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    negative_embeds: torch.Tensor | None,
    negative_mask: torch.Tensor | None,
) -> tuple[Image.Image, float]:
    """Generate with the unmodified official PixArt-Sigma transformer."""
    from diffusers import PixArtSigmaPipeline, PixArtTransformer2DModel

    base = PixArtTransformer2DModel.from_pretrained(
        args.transformer_model,
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).eval()
    pipe = PixArtSigmaPipeline.from_pretrained(
        args.component_model,
        transformer=base,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    kwargs: dict[str, Any] = {
        "prompt": None,
        "negative_prompt": None,
        "prompt_embeds": prompt_embeds.to("cuda"),
        "prompt_attention_mask": prompt_mask.to("cuda"),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": 512,
        "width": 512,
        "use_resolution_binning": False,
        "generator": torch.Generator("cuda").manual_seed(args.seed),
    }
    if negative_embeds is not None and negative_mask is not None:
        kwargs.update(
            negative_prompt_embeds=negative_embeds.to("cuda"),
            negative_prompt_attention_mask=negative_mask.to("cuda"),
        )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        image = pipe(**kwargs).images[0]
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    del pipe, base
    gc.collect()
    torch.cuda.empty_cache()
    return image, seconds


def validate_args(value: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if sys.version_info[:2] != (3, 11):
        value.error(f"Expected Python 3.11.x, got {platform.python_version()}.")
    if not torch.cuda.is_available() and not args.dry_run:
        value.error("A CUDA GPU is required.")
    if len(args.ranks) != 4 or len(set(args.ranks)) != 4:
        value.error("--ranks must have exactly four distinct entries.")
    if not args.prompt.strip():
        value.error("--prompt may not be empty.")
    if args.seed < 0 or args.num_inference_steps <= 0 or args.expected_groups <= 0:
        value.error("--seed, --num-inference-steps and --expected-groups must be positive.")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 1:
        value.error("--guidance-scale must be finite and at least 1.")


def main(argv: Sequence[str] | None = None) -> int:
    value = parser()
    args = value.parse_args(argv)
    validate_args(value, args)
    model_root, output_dir = args.model_root.resolve(), args.output_dir.resolve()
    groups = discover_checkpoint_groups(model_root, args.ranks, args.expected_groups)
    prompt_audit = audit_prompt(
        args.prompt, model_root, args.ranks, args.allow_seen_prompt
    )
    print(f"Discovered {len(groups)} checkpoint groups: {list(groups)}")
    print(f"Prompt exact-match audit: {prompt_audit['exact_training_caption_match']}")
    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_embeds, prompt_mask, negative_embeds, negative_mask, t5_seconds = encode_prompts(
        args, output_dir / "t5_offload"
    )
    individual_dir = output_dir / "individual"
    official_base_model: dict[str, Any] | None = None
    if args.include_base_model:
        print("Generating unmodified official PixArt-Sigma base model...")
        base_image, base_seconds = generate_official_base_image(
            args, prompt_embeds, prompt_mask, negative_embeds, negative_mask
        )
        base_path = output_dir / "official_base_model.png"
        base_image.save(base_path)
        official_base_model = {
            "transformer_model": args.transformer_model,
            "image": str(base_path),
            "generation_seconds": base_seconds,
        }
        print(f"Saved {base_path}")


    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step, checkpoints in groups.items():
        images: list[Image.Image] = []
        image_records: list[dict[str, Any]] = []
        for checkpoint in checkpoints:
            print(f"Generating step {step:,}, rank {checkpoint.rank}...")
            image, seconds = generate_image(
                checkpoint, args, prompt_embeds, prompt_mask, negative_embeds, negative_mask
            )
            image_path = individual_dir / f"step_{step:06d}_rank_{checkpoint.rank}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path)
            images.append(image)
            image_records.append(
                {
                    "rank": checkpoint.rank,
                    "adapter": str(checkpoint.adapter_dir),
                    "checkpoint_metadata": checkpoint.metadata,
                    "image": str(image_path),
                    "generation_seconds": seconds,
                }
            )
        grid_path = output_dir / f"step_{step:06d}_grid.png"
        build_grid(images, checkpoints, step, args, grid_path)
        records.append({"step": step, "grid": str(grid_path), "images": image_records})
        print(f"Saved {grid_path}")
    metadata = {
        "status": "PASS",
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "prompt_source_sample_id": args.prompt_source_sample_id,
        "prompt_audit": prompt_audit,
        "model_root": str(model_root),
        "ranks": list(args.ranks),
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "t5_encoding_seconds": t5_seconds,
        "official_base_model": official_base_model,
        "total_generation_seconds": time.perf_counter() - started,
        "runs": records,
    }
    metadata_path = output_dir / "evaluation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: wrote {len(records)} grids and {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

