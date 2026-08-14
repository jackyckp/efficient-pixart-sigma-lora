#!/usr/bin/env python3
"""Train a PixArt-Sigma LoRA from the project's precomputed local features.

The image VAE latents are read directly from the validated ZIP bundle. Prompt
embeddings are a separate, explicit input so that data preparation and GPU
training can be owned by different teammates without relying on row order.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import platform
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


TRANSFORMER_MODEL = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
COMPONENT_MODEL = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
EXPECTED_MANIFEST_FINGERPRINT = "b9d3c2d1d404"
EXPECTED_RESOLUTION = 512
EXPECTED_SCALING_FACTOR = 0.13025
EXPECTED_LATENT_SHAPE = (4, 64, 64)
EXPECTED_SEQUENCE_LENGTH = 300
EXPECTED_EMBEDDING_DIM = 4096
EXPECTED_DATASET_SIZE = 260
SUPPORTED_DATASET_SIZES = (50, 100, 209, 260)
EXPECTED_PAIRED_LATENT_CACHE = (
    "image_latents_n260_res512_b9d3c2d1d404.pt"
)
PROMPT_CACHE_KEYS = {
    "format_version",
    "sample_ids",
    "prompt_embeds",
    "attention_masks",
    "empty_prompt_embeds",
    "empty_prompt_attention_mask",
    "max_sequence_length",
    "text_encoder_model",
    "manifest_fingerprint",
}
OFFICIAL_TARGET_MODULES = [
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "proj_in",
    "proj_out",
    "ff.net.0.proj",
    "ff.net.2",
    "proj",
    "linear",
    "linear_1",
    "linear_2",
]


class AssetValidationError(ValueError):
    """Raised when an image, latent, or prompt asset violates its contract."""


class PromptCacheMissingError(FileNotFoundError):
    """Raised before model loading when the prompt cache is not available."""


@dataclass(frozen=True)
class LatentBundle:
    path: Path
    latents: torch.Tensor
    sample_ids: tuple[str, ...]
    manifest: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]
    validation_summary: dict[str, Any]
    latent_member: str
    manifest_member: str


@dataclass(frozen=True)
class PromptFeatures:
    path: Path
    sample_ids: tuple[str, ...]
    prompt_embeds: torch.Tensor
    attention_masks: torch.Tensor
    empty_prompt_embeds: torch.Tensor
    empty_prompt_attention_mask: torch.Tensor
    text_encoder_model: str
    manifest_fingerprint: str
    validation_summary_path: Path | None
    validation_summary: dict[str, Any] | None


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def manifest_fingerprint(manifest: Sequence[dict[str, Any]]) -> str:
    payload = [
        (
            row["sample_id"],
            normalize_newlines(row["caption"]),
            row["original_width"],
            row["original_height"],
        )
        for row in manifest
    ]
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:12]


def _validate_zip(archive: zipfile.ZipFile, path: Path) -> list[str]:
    names = archive.namelist()
    unsafe = []
    for name in names:
        member_path = PurePosixPath(name)
        if member_path.is_absolute() or ".." in member_path.parts:
            unsafe.append(name)
    if unsafe:
        raise AssetValidationError(
            f"{path} contains unsafe archive paths: {unsafe[:5]}"
        )
    corrupt_member = archive.testzip()
    if corrupt_member is not None:
        raise AssetValidationError(
            f"{path} is corrupted; first invalid member: {corrupt_member}"
        )
    return names


def _single_member(
    names: Sequence[str],
    *,
    suffix: str,
    filename: str | None = None,
) -> str:
    candidates = [
        name
        for name in names
        if not name.endswith("/")
        and name.endswith(suffix)
        and (filename is None or PurePosixPath(name).name == filename)
    ]
    if len(candidates) != 1:
        label = filename or f"*{suffix}"
        raise AssetValidationError(
            f"Expected exactly one {label} member; found {candidates}"
        )
    return candidates[0]


def _require_unique_strings(values: Any, label: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise AssetValidationError(f"{label} must be a non-empty list[str].")
    if len(values) != len(set(values)):
        duplicates = sorted(
            value for value in set(values) if values.count(value) > 1
        )
        raise AssetValidationError(
            f"{label} contains duplicate IDs: {duplicates[:10]}"
        )
    return tuple(values)


def load_latent_bundle(path: str | Path) -> LatentBundle:
    bundle_path = Path(path).resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Latent bundle does not exist: {bundle_path}")

    with zipfile.ZipFile(bundle_path) as archive:
        names = _validate_zip(archive, bundle_path)
        latent_member = _single_member(names, suffix=".pt")
        manifest_member = _single_member(
            names, suffix="manifest.jsonl", filename="manifest.jsonl"
        )
        summary_member = _single_member(
            names,
            suffix="validation_summary.json",
            filename="validation_summary.json",
        )
        manifest = tuple(
            json.loads(line)
            for line in archive.read(manifest_member)
            .decode("utf-8")
            .splitlines()
            if line.strip()
        )
        summary = json.loads(archive.read(summary_member))
        metadata = torch.load(
            BytesIO(archive.read(latent_member)),
            map_location="cpu",
            weights_only=True,
        )

    if not isinstance(metadata, dict):
        raise AssetValidationError("Latent cache must be a dictionary.")
    latents = metadata.get("latents")
    if not isinstance(latents, torch.Tensor):
        raise AssetValidationError("Latent cache key 'latents' must be a tensor.")
    if tuple(latents.shape) != (
        EXPECTED_DATASET_SIZE,
        *EXPECTED_LATENT_SHAPE,
    ):
        raise AssetValidationError(
            "Expected latent shape "
            f"[{EXPECTED_DATASET_SIZE}, {', '.join(map(str, EXPECTED_LATENT_SHAPE))}], "
            f"got {list(latents.shape)}."
        )
    if latents.dtype != torch.float16:
        raise AssetValidationError(
            f"Latents must be float16, got {latents.dtype}."
        )
    if not bool(torch.isfinite(latents).all()):
        raise AssetValidationError("Latents contain NaN or infinite values.")

    sample_ids = _require_unique_strings(
        metadata.get("sample_ids"), "latent sample_ids"
    )
    if len(sample_ids) != EXPECTED_DATASET_SIZE:
        raise AssetValidationError(
            f"Expected {EXPECTED_DATASET_SIZE} sample IDs, got {len(sample_ids)}."
        )
    if len(manifest) != EXPECTED_DATASET_SIZE:
        raise AssetValidationError(
            f"Expected {EXPECTED_DATASET_SIZE} manifest rows, got {len(manifest)}."
        )

    required_manifest_keys = {
        "sample_id",
        "relative_image_path",
        "relative_caption_path",
        "caption",
        "original_width",
        "original_height",
    }
    for index, row in enumerate(manifest):
        if not isinstance(row, dict) or not required_manifest_keys.issubset(row):
            raise AssetValidationError(
                f"Manifest row {index} is missing required keys."
            )
    manifest_ids = tuple(row["sample_id"] for row in manifest)
    if manifest_ids != sample_ids:
        raise AssetValidationError(
            "Manifest sample IDs do not exactly match latent sample IDs."
        )

    relative_paths = metadata.get("relative_image_paths")
    expected_paths = [row["relative_image_path"] for row in manifest]
    if relative_paths != expected_paths:
        raise AssetValidationError(
            "Manifest image paths do not exactly match latent metadata."
        )

    expected_metadata = {
        "format_version": 1,
        "num_images": EXPECTED_DATASET_SIZE,
        "resolution": EXPECTED_RESOLUTION,
        "latent_kind": "clean_x0_scaled",
        "vae_model": COMPONENT_MODEL,
        "transformer_model": TRANSFORMER_MODEL,
        "manifest_fingerprint": EXPECTED_MANIFEST_FINGERPRINT,
    }
    for key, expected in expected_metadata.items():
        actual = metadata.get(key)
        if actual != expected:
            raise AssetValidationError(
                f"Latent metadata {key!r}: expected {expected!r}, got {actual!r}."
            )
    scaling_factor = float(metadata.get("scaling_factor", math.nan))
    if not math.isclose(
        scaling_factor,
        EXPECTED_SCALING_FACTOR,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssetValidationError(
            f"Expected scaling factor {EXPECTED_SCALING_FACTOR}, "
            f"got {scaling_factor}."
        )

    actual_fingerprint = manifest_fingerprint(manifest)
    if actual_fingerprint != EXPECTED_MANIFEST_FINGERPRINT:
        raise AssetValidationError(
            f"Manifest fingerprint mismatch: expected "
            f"{EXPECTED_MANIFEST_FINGERPRINT}, got {actual_fingerprint}."
        )

    if summary.get("status") != "PASS":
        raise AssetValidationError(
            f"Latent validation summary is not PASS: {summary.get('status')!r}."
        )
    summary_expectations = {
        "num_images": EXPECTED_DATASET_SIZE,
        "latents_shape": [
            EXPECTED_DATASET_SIZE,
            *EXPECTED_LATENT_SHAPE,
        ],
        "latents_dtype": "torch.float16",
        "all_finite": True,
        "manifest_fingerprint": EXPECTED_MANIFEST_FINGERPRINT,
    }
    for key, expected in summary_expectations.items():
        if summary.get(key) != expected:
            raise AssetValidationError(
                f"Validation summary {key!r}: expected {expected!r}, "
                f"got {summary.get(key)!r}."
            )

    return LatentBundle(
        path=bundle_path,
        latents=latents.contiguous(),
        sample_ids=sample_ids,
        manifest=manifest,
        metadata=metadata,
        validation_summary=summary,
        latent_member=latent_member,
        manifest_member=manifest_member,
    )


def _natural_key(value: str) -> tuple[Any, ...]:
    import re

    parts = re.split(r"(\d+)", value.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def audit_image_archive(
    path: str | Path,
    latent_bundle: LatentBundle,
) -> dict[str, Any]:
    """Verify that ink.zip reproduces the latent bundle's canonical manifest."""

    from PIL import Image

    archive_path = Path(path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Image archive does not exist: {archive_path}")

    supported_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    with zipfile.ZipFile(archive_path) as archive:
        names = _validate_zip(archive, archive_path)
        image_members = sorted(
            (
                name
                for name in names
                if PurePosixPath(name).suffix.lower() in supported_extensions
            ),
            key=_natural_key,
        )
        text_members = {
            name
            for name in names
            if PurePosixPath(name).suffix.lower() == ".txt"
        }
        records: list[dict[str, Any]] = []
        categories: dict[str, int] = {}
        for image_member in image_members:
            member_path = PurePosixPath(image_member)
            if len(member_path.parts) < 3 or member_path.parts[0] != "ink":
                raise AssetValidationError(
                    f"Unexpected image archive path: {image_member}"
                )
            caption_member = member_path.with_suffix(".txt").as_posix()
            if caption_member not in text_members:
                raise AssetValidationError(
                    f"Missing caption for {image_member}."
                )
            relative_image = PurePosixPath(*member_path.parts[1:])
            relative_caption = relative_image.with_suffix(".txt")
            sample_id = relative_image.with_suffix("").as_posix()
            caption = normalize_newlines(
                archive.read(caption_member).decode("utf-8-sig")
            ).strip()
            if not caption:
                raise AssetValidationError(
                    f"Caption is empty: {caption_member}"
                )
            with Image.open(BytesIO(archive.read(image_member))) as image:
                width, height = image.size
                if image.mode != "RGB":
                    raise AssetValidationError(
                        f"{image_member} must be RGB, got {image.mode}."
                    )
            category = relative_image.parts[0]
            categories[category] = categories.get(category, 0) + 1
            records.append(
                {
                    "sample_id": sample_id,
                    "relative_image_path": relative_image.as_posix(),
                    "relative_caption_path": relative_caption.as_posix(),
                    "caption": caption,
                    "original_width": width,
                    "original_height": height,
                }
            )

        expected_caption_members = {
            PurePosixPath(image).with_suffix(".txt").as_posix()
            for image in image_members
        }
        orphan_captions = sorted(text_members - expected_caption_members)
        if orphan_captions:
            raise AssetValidationError(
                f"Found orphan captions: {orphan_captions[:10]}"
            )

    if tuple(records) != latent_bundle.manifest:
        for index, (actual, expected) in enumerate(
            zip(records, latent_bundle.manifest)
        ):
            if actual != expected:
                raise AssetValidationError(
                    f"Image archive differs from latent manifest at row "
                    f"{index}: {actual['sample_id']!r}."
                )
        raise AssetValidationError(
            "Image archive and latent manifest have different row counts."
        )

    fingerprint = manifest_fingerprint(records)
    if fingerprint != EXPECTED_MANIFEST_FINGERPRINT:
        raise AssetValidationError(
            f"Image archive fingerprint mismatch: {fingerprint}."
        )
    return {
        "num_images": len(records),
        "num_captions": len(text_members),
        "categories": categories,
        "manifest_fingerprint": fingerprint,
    }


def deterministic_subset_ids(
    sample_ids: Sequence[str],
    num_images: int,
    seed: int,
    category: str | None = None,
) -> tuple[str, ...]:
    pool = tuple(sample_ids)
    if category is not None:
        normalized_category = category.strip().strip("/")
        if not normalized_category:
            raise AssetValidationError("category may not be empty.")
        prefix = f"{normalized_category}/"
        pool = tuple(
            sample_id
            for sample_id in pool
            if sample_id.startswith(prefix)
        )
        if not pool:
            raise AssetValidationError(
                f"No samples found for category {normalized_category!r}."
            )
    if num_images <= 0 or num_images > len(pool):
        pool_label = (
            f"category {category!r}" if category is not None else "dataset"
        )
        raise AssetValidationError(
            f"num_images must be in [1, {len(pool)}] for {pool_label}, "
            f"got {num_images}."
        )
    ranked = sorted(
        pool,
        key=lambda sample_id: (
            hashlib.sha256(f"{seed}{sample_id}".encode("utf-8")).digest(),
            sample_id,
        ),
    )
    return tuple(ranked[:num_images])


def load_prompt_cache(
    path: str | Path,
    selected_sample_ids: Sequence[str],
    *,
    expected_fingerprint: str = EXPECTED_MANIFEST_FINGERPRINT,
    validation_summary_path: str | Path | None = None,
) -> PromptFeatures:
    cache_path = Path(path).resolve()
    if not cache_path.is_file():
        raise PromptCacheMissingError(
            "Prompt embedding cache is not available yet.\n"
            f"Expected path: {cache_path}\n"
            "Create the cache according to data/README.md, or run this script "
            "with --validate-assets-only."
        )

    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(cache, dict):
        raise AssetValidationError("Prompt cache must be a dictionary.")
    missing_keys = sorted(PROMPT_CACHE_KEYS - set(cache))
    if missing_keys:
        raise AssetValidationError(
            f"Prompt cache is missing keys: {missing_keys}"
        )
    if cache["format_version"] != 1:
        raise AssetValidationError(
            f"Prompt cache format_version must be 1, "
            f"got {cache['format_version']!r}."
        )

    sample_ids = _require_unique_strings(
        cache["sample_ids"], "prompt sample_ids"
    )
    prompt_embeds = cache["prompt_embeds"]
    attention_masks = cache["attention_masks"]
    empty_prompt_embeds = cache["empty_prompt_embeds"]
    empty_prompt_attention_mask = cache["empty_prompt_attention_mask"]
    if not isinstance(prompt_embeds, torch.Tensor):
        raise AssetValidationError("prompt_embeds must be a tensor.")
    if not isinstance(attention_masks, torch.Tensor):
        raise AssetValidationError("attention_masks must be a tensor.")
    expected_embed_shape = (
        len(sample_ids),
        EXPECTED_SEQUENCE_LENGTH,
        EXPECTED_EMBEDDING_DIM,
    )
    expected_mask_shape = (len(sample_ids), EXPECTED_SEQUENCE_LENGTH)
    if tuple(prompt_embeds.shape) != expected_embed_shape:
        raise AssetValidationError(
            f"prompt_embeds must have shape {expected_embed_shape}, "
            f"got {tuple(prompt_embeds.shape)}."
        )
    if tuple(attention_masks.shape) != expected_mask_shape:
        raise AssetValidationError(
            f"attention_masks must have shape {expected_mask_shape}, "
            f"got {tuple(attention_masks.shape)}."
        )
    if prompt_embeds.dtype != torch.float16:
        raise AssetValidationError(
            f"prompt_embeds must be float16, got {prompt_embeds.dtype}."
        )
    if attention_masks.dtype not in (torch.bool, torch.int64):
        raise AssetValidationError(
            "attention_masks must be bool or int64, "
            f"got {attention_masks.dtype}."
        )
    if not bool(torch.isfinite(prompt_embeds).all()):
        raise AssetValidationError(
            "prompt_embeds contains NaN or infinite values."
        )
    if not bool(((attention_masks == 0) | (attention_masks == 1)).all()):
        raise AssetValidationError(
            "attention_masks may only contain 0/1 values."
        )
    expected_empty_embed_shape = (
        1,
        EXPECTED_SEQUENCE_LENGTH,
        EXPECTED_EMBEDDING_DIM,
    )
    expected_empty_mask_shape = (1, EXPECTED_SEQUENCE_LENGTH)
    if not isinstance(empty_prompt_embeds, torch.Tensor) or tuple(
        empty_prompt_embeds.shape
    ) != expected_empty_embed_shape:
        raise AssetValidationError(
            "empty_prompt_embeds must be a tensor with shape "
            f"{expected_empty_embed_shape}."
        )
    if empty_prompt_embeds.dtype != torch.float16 or not bool(
        torch.isfinite(empty_prompt_embeds).all()
    ):
        raise AssetValidationError(
            "empty_prompt_embeds must be finite float16."
        )
    if not isinstance(empty_prompt_attention_mask, torch.Tensor) or tuple(
        empty_prompt_attention_mask.shape
    ) != expected_empty_mask_shape:
        raise AssetValidationError(
            "empty_prompt_attention_mask must be a tensor with shape "
            f"{expected_empty_mask_shape}."
        )
    if empty_prompt_attention_mask.dtype not in (torch.bool, torch.int64):
        raise AssetValidationError(
            "empty_prompt_attention_mask must be bool or int64."
        )
    if not bool(
        (
            (empty_prompt_attention_mask == 0)
            | (empty_prompt_attention_mask == 1)
        ).all()
    ):
        raise AssetValidationError(
            "empty_prompt_attention_mask may only contain 0/1 values."
        )
    if cache["max_sequence_length"] != EXPECTED_SEQUENCE_LENGTH:
        raise AssetValidationError(
            f"max_sequence_length must be {EXPECTED_SEQUENCE_LENGTH}, "
            f"got {cache['max_sequence_length']!r}."
        )
    if cache["manifest_fingerprint"] != expected_fingerprint:
        raise AssetValidationError(
            f"Prompt manifest fingerprint must be {expected_fingerprint}, "
            f"got {cache['manifest_fingerprint']!r}."
        )
    text_encoder_model = cache["text_encoder_model"]
    if not isinstance(text_encoder_model, str) or not text_encoder_model:
        raise AssetValidationError(
            "text_encoder_model must be a non-empty string."
        )

    validation_path: Path | None = None
    validation_summary: dict[str, Any] | None = None
    if validation_summary_path is not None:
        validation_path = Path(validation_summary_path).resolve()
        if not validation_path.is_file():
            raise FileNotFoundError(
                "Prompt validation summary does not exist: "
                f"{validation_path}"
            )
        validation_summary = json.loads(
            validation_path.read_text(encoding="utf-8")
        )
        if not isinstance(validation_summary, dict):
            raise AssetValidationError(
                "Prompt validation summary must be a JSON object."
            )
        summary_expectations = {
            "status": "PASS",
            "cache_file": cache_path.name,
            "num_samples": len(sample_ids),
            "prompt_embeds_shape": list(expected_embed_shape),
            "prompt_embeds_dtype": "torch.float16",
            "attention_masks_shape": list(expected_mask_shape),
            "attention_masks_dtype": str(attention_masks.dtype),
            "all_finite": True,
            "manifest_fingerprint": expected_fingerprint,
            "text_encoder_model": text_encoder_model,
            "transformer_model": TRANSFORMER_MODEL,
            "paired_clean_latent_cache": EXPECTED_PAIRED_LATENT_CACHE,
        }
        for key, expected in summary_expectations.items():
            actual = validation_summary.get(key)
            if actual != expected:
                raise AssetValidationError(
                    f"Prompt validation summary {key!r}: expected "
                    f"{expected!r}, got {actual!r}."
                )
    index_by_id = {
        sample_id: index for index, sample_id in enumerate(sample_ids)
    }
    missing_ids = [
        sample_id
        for sample_id in selected_sample_ids
        if sample_id not in index_by_id
    ]
    if missing_ids:
        raise AssetValidationError(
            f"Prompt cache does not cover {len(missing_ids)} selected IDs: "
            f"{missing_ids[:10]}"
        )
    selected_indices = torch.tensor(
        [index_by_id[sample_id] for sample_id in selected_sample_ids],
        dtype=torch.long,
    )
    return PromptFeatures(
        path=cache_path,
        sample_ids=tuple(selected_sample_ids),
        prompt_embeds=prompt_embeds.index_select(
            0, selected_indices
        ).contiguous(),
        attention_masks=attention_masks.index_select(
            0, selected_indices
        ).contiguous(),
        empty_prompt_embeds=empty_prompt_embeds.contiguous(),
        empty_prompt_attention_mask=empty_prompt_attention_mask.contiguous(),
        text_encoder_model=text_encoder_model,
        manifest_fingerprint=cache["manifest_fingerprint"],
        validation_summary_path=validation_path,
        validation_summary=validation_summary,
    )


class CachedLocalDataset(Dataset):
    def __init__(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        attention_masks: torch.Tensor,
    ) -> None:
        if not (
            len(latents) == len(prompt_embeds) == len(attention_masks)
        ):
            raise AssetValidationError(
                "Latent and prompt feature row counts do not match."
            )
        self.latents = latents
        self.prompt_embeds = prompt_embeds
        self.attention_masks = attention_masks

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "latents": self.latents[index],
            "prompt_embeds": self.prompt_embeds[index],
            "attention_mask": self.attention_masks[index],
        }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _save_lora_checkpoint(
    *,
    accelerator: Any,
    transformer: Any,
    output_dir: Path,
    args: argparse.Namespace,
    global_step: int,
    loss_value: float,
) -> str:
    checkpoint_dir = (
        output_dir / "checkpoints" / f"step_{global_step:06d}"
    )
    checkpoint_adapter_dir = checkpoint_dir / "lora_adapter"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_adapter_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(transformer).save_pretrained(
            checkpoint_adapter_dir,
            safe_serialization=True,
        )
        checkpoint_metadata = {
            "status": "CHECKPOINT",
            "optimizer_step": global_step,
            "loss": loss_value,
            "rank": args.rank,
            "lora_alpha": args.lora_alpha,
            "learning_rate": args.learning_rate,
            "num_images": args.num_images,
            "dataset_category": args.category,
            "seed": args.seed,
            "inference_steps": args.inference_steps,
            "guidance_scale": args.guidance_scale,
            "manifest_fingerprint": EXPECTED_MANIFEST_FINGERPRINT,
            "adapter": str(checkpoint_adapter_dir),
        }
        _write_json(
            checkpoint_dir / "checkpoint_metadata.json",
            checkpoint_metadata,
        )
        _write_json(
            output_dir / "checkpoints" / "latest_checkpoint.json",
            checkpoint_metadata,
        )
    accelerator.wait_for_everyone()
    print(f"CHECKPOINT: step={global_step} path={checkpoint_dir}")
    return str(checkpoint_dir)


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def run_training(
    args: argparse.Namespace,
    latent_bundle: LatentBundle,
    selected_ids: tuple[str, ...],
    prompt_features: PromptFeatures,
) -> dict[str, Any]:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"Expected Python 3.11.x, got {platform.python_version()}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for PixArt LoRA training.")

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import (
        DDPMScheduler,
        PixArtSigmaPipeline,
        PixArtTransformer2DModel,
    )
    from peft import LoraConfig, PeftModel, get_peft_model

    set_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    latent_index = {
        sample_id: index
        for index, sample_id in enumerate(latent_bundle.sample_ids)
    }
    selected_indices = torch.tensor(
        [latent_index[sample_id] for sample_id in selected_ids],
        dtype=torch.long,
    )
    selected_latents = latent_bundle.latents.index_select(
        0, selected_indices
    ).contiguous()
    manifest_by_id = {
        row["sample_id"]: row for row in latent_bundle.manifest
    }
    selected_manifest = [manifest_by_id[sample_id] for sample_id in selected_ids]

    output_dir = Path(args.output_dir).resolve()
    adapter_dir = output_dir / "lora_adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "subset_manifest.json", selected_manifest)

    dataset = CachedLocalDataset(
        selected_latents,
        prompt_features.prompt_embeds,
        prompt_features.attention_masks,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=loader_generator,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.component_model,
        subfolder="scheduler",
    )
    transformer = PixArtTransformer2DModel.from_pretrained(
        args.transformer_model,
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    transformer = get_peft_model(
        transformer,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=OFFICIAL_TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
        ),
    )
    for parameter in transformer.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)

    trainable_names = [
        name
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names or not all(
        "lora_" in name for name in trainable_names
    ):
        raise RuntimeError("Unexpected trainable parameters after LoRA setup.")
    if (
        transformer.config.in_channels != 4
        or transformer.config.out_channels != 8
    ):
        raise RuntimeError(
            "Expected PixArt transformer channels 4 -> 8, got "
            f"{transformer.config.in_channels} -> "
            f"{transformer.config.out_channels}."
        )

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in transformer.parameters()
            if parameter.requires_grad
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )
    transformer, optimizer, train_loader = accelerator.prepare(
        transformer,
        optimizer,
        train_loader,
    )

    global_step = 0
    loss_history: list[float] = []
    checkpoint_paths: list[str] = []
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    transformer.train()

    while global_step < args.max_train_steps:
        for batch in train_loader:
            with accelerator.accumulate(transformer):
                latents = batch["latents"].to(
                    accelerator.device,
                    dtype=torch.float16,
                    non_blocking=True,
                )
                prompt_embeds = batch["prompt_embeds"].to(
                    accelerator.device,
                    dtype=torch.float16,
                    non_blocking=True,
                )
                attention_mask = batch["attention_mask"].to(
                    accelerator.device,
                    non_blocking=True,
                )
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(
                    latents,
                    noise,
                    timesteps,
                )
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif (
                    noise_scheduler.config.prediction_type
                    == "v_prediction"
                ):
                    target = noise_scheduler.get_velocity(
                        latents,
                        noise,
                        timesteps,
                    )
                else:
                    raise RuntimeError(
                        "Unsupported scheduler prediction type: "
                        f"{noise_scheduler.config.prediction_type}"
                    )

                with accelerator.autocast():
                    model_output = transformer(
                        noisy_latents,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=attention_mask,
                        timestep=timesteps,
                        added_cond_kwargs={
                            "resolution": None,
                            "aspect_ratio": None,
                        },
                    ).sample
                    if model_output.shape[1] != target.shape[1] * 2:
                        raise RuntimeError(
                            "Expected PixArt noise + learned-sigma output."
                        )
                    model_prediction = model_output.chunk(2, dim=1)[0]
                    loss = F.mse_loss(
                        model_prediction.float(),
                        target.float(),
                        reduction="mean",
                    )

                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite loss at optimizer step {global_step}: "
                        f"{loss.item()}"
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        transformer.parameters(),
                        args.max_grad_norm,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                loss_value = (
                    accelerator.gather(loss.detach()).mean().item()
                )
                loss_history.append(loss_value)
                print(
                    f"optimizer_step={global_step:02d}/"
                    f"{args.max_train_steps} loss={loss_value:.6f}"
                )
                if (
                    args.checkpointing_steps > 0
                    and global_step % args.checkpointing_steps == 0
                ):
                    ckpt_root = output_dir / f"checkpoint-{global_step}"
                    ckpt_adapter = ckpt_root / "lora_adapter"
                    ckpt_adapter.mkdir(parents=True, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(transformer)
                    unwrapped.save_pretrained(ckpt_adapter, safe_serialization=True)
                    (ckpt_root / "subset_manifest.json").write_text(
                        json.dumps(selected_manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    print(f"Saved checkpoint at step {global_step} to {ckpt_root}")
            if global_step >= args.max_train_steps:
                break

    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    if (
        global_step != args.max_train_steps
        or len(loss_history) != args.max_train_steps
        or not all(math.isfinite(value) for value in loss_history)
    ):
        raise RuntimeError("Training did not complete all finite updates.")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        unwrapped_transformer.save_pretrained(
            adapter_dir,
            safe_serialization=True,
        )
    accelerator.wait_for_everyone()

    adapter_config = adapter_dir / "adapter_config.json"
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_config.is_file() or not adapter_weights.is_file():
        raise RuntimeError("PEFT adapter save did not produce expected files.")

    del optimizer, train_loader, transformer, accelerator
    gc.collect()
    torch.cuda.empty_cache()

    base_transformer = PixArtTransformer2DModel.from_pretrained(
        args.transformer_model,
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    reloaded_transformer = PeftModel.from_pretrained(
        base_transformer,
        adapter_dir,
        is_trainable=False,
    ).eval()
    loaded_rank = reloaded_transformer.peft_config["default"].r
    if loaded_rank != args.rank:
        raise RuntimeError(
            f"Reloaded LoRA rank {loaded_rank} != expected {args.rank}."
        )
    if any(
        parameter.requires_grad
        for parameter in reloaded_transformer.parameters()
    ):
        raise RuntimeError("Reloaded inference adapter is unexpectedly trainable.")
    pipeline_transformer = reloaded_transformer.merge_and_unload().eval()
    if not isinstance(pipeline_transformer, PixArtTransformer2DModel):
        raise RuntimeError("PEFT reload did not expose a PixArt transformer.")

    pipeline = PixArtSigmaPipeline.from_pretrained(
        args.component_model,
        transformer=pipeline_transformer,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    inference_embeds = prompt_features.prompt_embeds[:1].to(
        "cuda",
        dtype=torch.float16,
    )
    inference_mask = prompt_features.attention_masks[:1].to("cuda")
    generation_kwargs: dict[str, Any] = {
        "prompt": None,
        "negative_prompt": None,
        "prompt_embeds": inference_embeds,
        "prompt_attention_mask": inference_mask,
        "num_inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": EXPECTED_RESOLUTION,
        "width": EXPECTED_RESOLUTION,
        "use_resolution_binning": False,
    }
    if args.guidance_scale > 1.0:
        generation_kwargs.update(
            negative_prompt_embeds=(
                prompt_features.empty_prompt_embeds.to("cuda")
            ),
            negative_prompt_attention_mask=(
                prompt_features.empty_prompt_attention_mask.to("cuda")
            ),
        )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    generation_kwargs["generator"] = generator
    torch.cuda.synchronize()
    inference_start = time.perf_counter()
    with torch.inference_mode():
        image = pipeline(**generation_kwargs).images[0]
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    generated_path = output_dir / "reload_generation.png"
    image.save(generated_path)
    if image.size != (EXPECTED_RESOLUTION, EXPECTED_RESOLUTION):
        raise RuntimeError(f"Unexpected generated image size: {image.size}.")

    run_metadata = {
        "status": "PASS",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "diffusers": _package_version("diffusers"),
        "transformers": _package_version("transformers"),
        "accelerate": _package_version("accelerate"),
        "peft": _package_version("peft"),
        "transformer_model": args.transformer_model,
        "component_model": args.component_model,
        "latent_bundle": str(latent_bundle.path),
        "latent_member": latent_bundle.latent_member,
        "manifest_fingerprint": EXPECTED_MANIFEST_FINGERPRINT,
        "prompt_cache": str(prompt_features.path),
        "prompt_validation_summary": (
            str(prompt_features.validation_summary_path)
            if prompt_features.validation_summary_path
            else None
        ),
        "prompt_validation_status": (
            prompt_features.validation_summary.get("status")
            if prompt_features.validation_summary
            else None
        ),
        "prompt_text_encoder_model": prompt_features.text_encoder_model,
        "prompt_manifest_fingerprint": (
            prompt_features.manifest_fingerprint
        ),
        "run_role": args.run_role,
        "dataset_category": args.category,
        "num_images": len(selected_ids),
        "selected_sample_ids": list(selected_ids),
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "optimizer_steps": global_step,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "loss_history": loss_history,
        "train_seconds": train_seconds,
        "seconds_per_optimizer_step": train_seconds / global_step,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "checkpoints": checkpoint_paths,
        "peak_allocated_vram_gb": peak_vram_gb,
        "inference_sample_id": selected_ids[0],
        "inference_prompt": selected_manifest[0]["caption"],
        "inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "inference_seconds": inference_seconds,
        "generated_image": str(generated_path),
    }
    _write_json(output_dir / "run_metadata.json", run_metadata)
    print(
        "PASS: train -> save -> fresh reload -> generate\n"
        f"Output: {output_dir}\n"
        f"Training: {train_seconds:.2f}s; "
        f"inference: {inference_seconds:.2f}s; "
        f"peak allocated VRAM: {peak_vram_gb:.2f} GiB"
    )
    return run_metadata


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description=(
            "Validate or train PixArt-Sigma LoRA from the project's local "
            "latent and prompt feature caches."
        )
    )
    parser.add_argument(
        "--latent-bundle",
        type=Path,
        default=root / "data" / "archives" / "clean_latents_512.zip",
    )
    parser.add_argument(
        "--image-archive",
        type=Path,
        default=(
            root / "data" / "ink.zip"
            if (root / "data" / "ink.zip").is_file()
            else root / "data" / "archives" / "ink.zip"
        ),
    )
    parser.add_argument(
        "--prompt-cache",
        type=Path,
        default=(
            root
            / "data"
            / "features"
            / "t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt"
        ),
    )
    parser.add_argument(
        "--prompt-validation-summary",
        type=Path,
        default=(
            root
            / "data"
            / "features"
            / "validation_summary.json"
        ),
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help=(
            "Number of deterministic samples after optional category "
            "filtering."
        ),
    )
    parser.add_argument(
        "--category",
        default=None,
        help=(
            "Optional sample-ID category prefix, for example 'plant'. "
            "The current canonical archive contains 209 plant samples."
        ),
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--mixed-precision",
        choices=("no", "fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--run-role", default="lora_smoke")
    parser.add_argument(
        "--checkpointing-steps",
        type=int,
        default=0,
        help="Save intermediate PEFT adapter checkpoints every N steps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--transformer-model",
        default=TRANSFORMER_MODEL,
    )
    parser.add_argument(
        "--component-model",
        default=COMPONENT_MODEL,
    )
    parser.add_argument(
        "--plant-only",
        action="store_true",
        help="Filter training subset exclusively to plant images (209 samples).",
    )
    parser.add_argument(
        "--validate-assets-only",
        action="store_true",
        help=(
            "Validate the image/latent bundles and, if present, the prompt "
            "cache without downloading models or starting training."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.rank <= 0:
        parser.error("--rank must be positive.")
    if args.max_train_steps <= 0:
        parser.error("--max-train-steps must be positive.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive.")
    if args.train_batch_size <= 0:
        parser.error("--train-batch-size must be positive.")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be positive.")
    if args.inference_steps <= 0:
        parser.error("--inference-steps must be positive.")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 1.0:
        parser.error("--guidance-scale must be finite and at least 1.0.")
    if args.checkpoint_every_steps < 0:
        parser.error("--checkpoint-every-steps may not be negative.")
    if (
        args.checkpoint_every_steps > args.max_train_steps
    ):
        parser.error("--checkpoint-every-steps exceeds training length.")
    if args.log_every_steps <= 0:
        parser.error("--log-every-steps must be positive.")
    if args.lora_alpha is None:
        args.lora_alpha = args.rank
    if args.output_dir is None:
        suffix = "plant209" if args.plant_only or args.num_images == 209 else f"n{args.num_images}"
        args.output_dir = (
            repository_root()
            / "outputs"
            / "plant_dataset"
            / f"r{args.rank}_{suffix}"
        )

    latent_bundle = load_latent_bundle(args.latent_bundle)
    image_audit = audit_image_archive(
        args.image_archive,
        latent_bundle,
    )
    if args.plant_only or args.num_images == 209:
        candidate_ids = tuple(s for s in latent_bundle.sample_ids if s.startswith("plant/"))
    else:
        candidate_ids = latent_bundle.sample_ids

    target_count = len(candidate_ids) if (args.plant_only or args.num_images == 209) and args.num_images >= len(candidate_ids) else args.num_images
    selected_ids = deterministic_subset_ids(
        candidate_ids,
        target_count,
        args.seed,
        args.category,
    )
    print(
        "PASS: local image and latent assets\n"
        f"Images: {image_audit['num_images']}; "
        f"categories: {image_audit['categories']}\n"
        f"Latents: {list(latent_bundle.latents.shape)} "
        f"{latent_bundle.latents.dtype}; "
        f"fingerprint: {EXPECTED_MANIFEST_FINGERPRINT}\n"
        f"Selected deterministic subset: {len(selected_ids)} samples; "
        f"category: {args.category or 'all'}"
    )

    if args.validate_assets_only:
        if Path(args.prompt_cache).is_file():
            prompt_features = load_prompt_cache(
                args.prompt_cache,
                selected_ids,
                validation_summary_path=args.prompt_validation_summary,
            )
            print(
                "PASS: prompt cache covers selected subset "
                f"({len(prompt_features.sample_ids)} rows)."
            )
        else:
            print(
                "PENDING: prompt embedding cache is not present yet.\n"
                f"Expected future path: {Path(args.prompt_cache).resolve()}\n"
                "The image and latent assets are ready."
            )
        return 0

    try:
        prompt_features = load_prompt_cache(
            args.prompt_cache,
            selected_ids,
            validation_summary_path=args.prompt_validation_summary,
        )
    except PromptCacheMissingError as exc:
        parser.error(str(exc))
    run_training(
        args,
        latent_bundle,
        selected_ids,
        prompt_features,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
