#!/usr/bin/env python3
"""Shared contracts and math for PixArt phased trajectory distillation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


TRANSFORMER_MODEL = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
COMPONENT_MODEL = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
MANIFEST_FINGERPRINT = "b9d3c2d1d404"
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 4096
LATENT_SHAPE = (4, 64, 64)
EXPECTED_RANK = 16
EXPECTED_ALPHA = 16
EXPECTED_ADAPTER_TENSORS = 574
EXPECTED_ADAPTER_PARAMETERS = 13_765_376
TEACHER_TIMESTEPS = (
    999,
    949,
    899,
    849,
    799,
    749,
    699,
    649,
    599,
    549,
    500,
    450,
    400,
    350,
    300,
    250,
    200,
    150,
    100,
    50,
)
PHASE_INDEX_PAIRS = {
    4: ((0, 5), (5, 10), (10, 15), (15, 20)),
    2: ((0, 10), (10, 20)),
}
OFFICIAL_TARGET_MODULES = frozenset(
    {
        "proj_in",
        "to_k",
        "proj_out",
        "proj",
        "to_out.0",
        "linear_2",
        "to_q",
        "ff.net.2",
        "linear_1",
        "linear",
        "to_v",
        "ff.net.0.proj",
    }
)
STYLE_TRIGGER = "traditional Chinese ink wash painting style, shuimo hua"
SUBJECT_SUFFIX = f"minimal composition, {STYLE_TRIGGER}"
FULL_SUFFIX = (
    "balanced negative space, expressive wet and dry brushwork, "
    f"{STYLE_TRIGGER}"
)


class DistillationContractError(ValueError):
    """Raised when a distillation asset violates its public contract."""


@dataclass(frozen=True)
class PromptBank:
    path: Path
    records: tuple[dict[str, Any], ...]
    fingerprint: str


@dataclass(frozen=True)
class DistillPromptFeatures:
    path: Path
    prompt_ids: tuple[str, ...]
    source_sample_ids: tuple[str, ...]
    prompt_embeds: torch.Tensor
    attention_masks: torch.Tensor
    empty_prompt_embeds: torch.Tensor
    empty_prompt_attention_mask: torch.Tensor
    prompt_bank_fingerprint: str
    text_encoder_model: str


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_records(records: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(canonical_json(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def resolve_adapter_dir(path: str | Path) -> Path:
    candidate = Path(path).resolve()
    direct = candidate / "adapter_model.safetensors"
    nested = candidate / "lora_adapter" / "adapter_model.safetensors"
    if direct.is_file() and (candidate / "adapter_config.json").is_file():
        return candidate
    if nested.is_file() and (
        candidate / "lora_adapter" / "adapter_config.json"
    ).is_file():
        return candidate / "lora_adapter"
    raise FileNotFoundError(
        "No PEFT adapter found. Expected adapter_config.json and "
        f"adapter_model.safetensors in {candidate} or its lora_adapter/."
    )


def load_adapter_config(adapter_dir: str | Path) -> dict[str, Any]:
    path = resolve_adapter_dir(adapter_dir) / "adapter_config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistillationContractError(
            f"Invalid adapter config JSON: {path}"
        ) from error
    if not isinstance(config, dict):
        raise DistillationContractError("Adapter config must be an object.")
    return config


def inspect_adapter(adapter: str | Path) -> dict[str, Any]:
    from safetensors import safe_open

    adapter_dir = resolve_adapter_dir(adapter)
    config = load_adapter_config(adapter_dir)
    if config.get("peft_type") != "LORA":
        raise DistillationContractError("Teacher adapter must be PEFT LoRA.")
    if config.get("r") != EXPECTED_RANK:
        raise DistillationContractError(
            f"Teacher rank must be {EXPECTED_RANK}, got {config.get('r')!r}."
        )
    if config.get("lora_alpha") != EXPECTED_ALPHA:
        raise DistillationContractError(
            "Teacher lora_alpha must be "
            f"{EXPECTED_ALPHA}, got {config.get('lora_alpha')!r}."
        )
    target_modules = frozenset(config.get("target_modules") or ())
    if target_modules != OFFICIAL_TARGET_MODULES:
        raise DistillationContractError(
            "Teacher target_modules do not match the official local trainer."
        )

    weights_path = adapter_dir / "adapter_model.safetensors"
    tensor_count = 0
    parameter_count = 0
    dtypes: set[str] = set()
    shapes: dict[str, list[int]] = {}
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for key in keys:
            tensor = handle.get_tensor(key)
            tensor_count += 1
            parameter_count += tensor.numel()
            dtypes.add(str(tensor.dtype))
            shapes[key] = list(tensor.shape)
            if not bool(torch.isfinite(tensor).all()):
                raise DistillationContractError(
                    f"Teacher tensor contains NaN or infinity: {key}"
                )
    if tensor_count != EXPECTED_ADAPTER_TENSORS:
        raise DistillationContractError(
            f"Expected {EXPECTED_ADAPTER_TENSORS} adapter tensors, "
            f"got {tensor_count}."
        )
    if parameter_count != EXPECTED_ADAPTER_PARAMETERS:
        raise DistillationContractError(
            f"Expected {EXPECTED_ADAPTER_PARAMETERS} adapter parameters, "
            f"got {parameter_count}."
        )
    if dtypes != {"torch.float32"}:
        raise DistillationContractError(
            f"Teacher adapter tensors must all be float32, got {sorted(dtypes)}."
        )
    return {
        "adapter_dir": str(adapter_dir),
        "adapter_weights": str(weights_path),
        "adapter_sha256": sha256_file(weights_path),
        "adapter_config_sha256": sha256_file(
            adapter_dir / "adapter_config.json"
        ),
        "rank": config["r"],
        "lora_alpha": config["lora_alpha"],
        "target_modules": sorted(target_modules),
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "dtypes": sorted(dtypes),
        "tensor_shapes_sha256": hashlib.sha256(
            canonical_json(shapes).encode("utf-8")
        ).hexdigest(),
    }


def strip_style_suffix(caption: str) -> str:
    text = " ".join(caption.split()).strip(" ,.")
    patterns = (
        r",?\s*Chinese ink wash painting style,?\s*Sumi-e\s*$",
        r",?\s*traditional Chinese ink wash painting style,?\s*shuimo hua\s*$",
    )
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip(" ,.")
    return text


def first_sentence(caption: str) -> str:
    clean = strip_style_suffix(caption)
    match = re.search(r"(?<=[.!?])\s+", clean)
    sentence = clean[: match.start() + 1] if match else clean
    return sentence.strip(" ,.")


def build_prompt_records(
    plant_manifest: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for row in sorted(plant_manifest, key=lambda item: item["sample_id"]):
        source_id = row["sample_id"]
        caption = " ".join(row["caption"].split())
        base = strip_style_suffix(caption)
        variants = (
            ("original", caption),
            ("subject", f"{first_sentence(caption)}, {SUBJECT_SUFFIX}"),
            ("styled", f"{base}, {FULL_SUFFIX}"),
        )
        for variant, prompt in variants:
            records.append(
                {
                    "prompt_id": f"{source_id}::{variant}",
                    "source_sample_id": source_id,
                    "variant": variant,
                    "prompt": prompt,
                    "category": "plant",
                    "training_only": True,
                }
            )
    prompt_ids = [row["prompt_id"] for row in records]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise DistillationContractError("Prompt bank IDs are not unique.")
    return tuple(records)


def save_prompt_bank(path: str | Path, records: Sequence[dict[str, Any]]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(canonical_json(record) + "\n" for record in records)
    output.write_text(text, encoding="utf-8", newline="\n")
    return fingerprint_records(records)


def load_prompt_bank(path: str | Path) -> PromptBank:
    bank_path = Path(path).resolve()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        bank_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise DistillationContractError(
                f"Invalid prompt bank JSON at line {line_number}."
            ) from error
        if not isinstance(record, dict):
            raise DistillationContractError(
                f"Prompt bank line {line_number} is not an object."
            )
        records.append(record)
    required = {
        "prompt_id",
        "source_sample_id",
        "variant",
        "prompt",
        "category",
    }
    if any(not required.issubset(record) for record in records):
        raise DistillationContractError("Prompt bank record is incomplete.")
    ids = [record["prompt_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise DistillationContractError("Prompt bank IDs are duplicated.")
    return PromptBank(
        path=bank_path,
        records=tuple(records),
        fingerprint=fingerprint_records(records),
    )


def deterministic_trajectory_seed(prompt_id: str, replica: int) -> int:
    if replica < 0:
        raise ValueError("replica must be non-negative.")
    digest = hashlib.sha256(
        f"pixart-distill-v1\0{prompt_id}\0{replica}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def phase_pairs(target_steps: int) -> tuple[tuple[int, int], ...]:
    try:
        return PHASE_INDEX_PAIRS[target_steps]
    except KeyError as error:
        raise ValueError("target_steps must be 2 or 4.") from error


def state_timestep(state_index: int) -> int:
    if state_index == len(TEACHER_TIMESTEPS):
        return -1
    if not 0 <= state_index < len(TEACHER_TIMESTEPS):
        raise IndexError(f"Invalid trajectory state index: {state_index}")
    return TEACHER_TIMESTEPS[state_index]


def _extract_alpha(
    alphas_cumprod: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
) -> torch.Tensor:
    if timestep == -1:
        return torch.ones((), dtype=torch.float32, device=sample.device)
    if timestep < 0 or timestep >= len(alphas_cumprod):
        raise ValueError(f"Invalid diffusion timestep: {timestep}")
    return alphas_cumprod[timestep].to(sample.device, dtype=torch.float32)


def deterministic_jump(
    sample: torch.Tensor,
    epsilon: torch.Tensor,
    start_timestep: int,
    target_timestep: int,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Apply one deterministic DDIM-form jump between arbitrary timesteps."""
    if target_timestep >= start_timestep:
        raise ValueError("target_timestep must be earlier than start_timestep.")
    if sample.shape != epsilon.shape:
        raise ValueError("sample and epsilon shapes must match.")
    sample_f = sample.float()
    epsilon_f = epsilon.float()
    alpha_start = _extract_alpha(alphas_cumprod, start_timestep, sample_f)
    alpha_target = _extract_alpha(alphas_cumprod, target_timestep, sample_f)
    beta_start = 1.0 - alpha_start
    beta_target = 1.0 - alpha_target
    predicted_origin = (
        sample_f - beta_start.sqrt() * epsilon_f
    ) / alpha_start.sqrt()
    return (
        alpha_target.sqrt() * predicted_origin
        + beta_target.sqrt() * epsilon_f
    )


def effective_epsilon_target(
    sample: torch.Tensor,
    target: torch.Tensor,
    start_timestep: int,
    target_timestep: int,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Invert deterministic_jump to obtain the effective epsilon target."""
    if sample.shape != target.shape:
        raise ValueError("sample and target shapes must match.")
    sample_f = sample.float()
    target_f = target.float()
    alpha_start = _extract_alpha(alphas_cumprod, start_timestep, sample_f)
    alpha_target = _extract_alpha(alphas_cumprod, target_timestep, sample_f)
    scale = (alpha_target / alpha_start).sqrt()
    coefficient = (1.0 - alpha_target).sqrt() - scale * (
        1.0 - alpha_start
    ).sqrt()
    if bool(torch.isclose(coefficient, torch.zeros_like(coefficient))):
        raise ValueError("Degenerate jump has zero epsilon coefficient.")
    return (target_f - scale * sample_f) / coefficient


def pseudo_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    c: float = 0.001,
) -> torch.Tensor:
    if not math.isfinite(c) or c <= 0:
        raise ValueError("Pseudo-Huber c must be finite and positive.")
    residual = prediction.float() - target.float()
    return (torch.sqrt(residual.square() + c * c) - c).mean()


def split_epsilon_prediction(
    model_output: torch.Tensor,
    latent_channels: int = 4,
) -> torch.Tensor:
    if model_output.ndim != 4 or model_output.shape[1] != latent_channels * 2:
        raise DistillationContractError(
            "Expected PixArt learned-sigma output with "
            f"{latent_channels * 2} channels, got {tuple(model_output.shape)}."
        )
    return model_output[:, :latent_channels]


def validate_prompt_feature_tensors(
    prompt_ids: Sequence[str],
    source_ids: Sequence[str],
    prompt_embeds: torch.Tensor,
    attention_masks: torch.Tensor,
) -> None:
    count = len(prompt_ids)
    if len(source_ids) != count or len(set(prompt_ids)) != count:
        raise DistillationContractError(
            "Prompt IDs/source IDs are inconsistent or duplicated."
        )
    if prompt_embeds.shape != (count, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
        raise DistillationContractError(
            "prompt_embeds must have shape "
            f"[{count}, {MAX_SEQUENCE_LENGTH}, {EMBEDDING_DIM}]."
        )
    if prompt_embeds.dtype != torch.float16 or not bool(
        torch.isfinite(prompt_embeds).all()
    ):
        raise DistillationContractError(
            "prompt_embeds must be finite CPU-loadable float16."
        )
    if attention_masks.shape != (count, MAX_SEQUENCE_LENGTH):
        raise DistillationContractError(
            f"attention_masks must have shape [{count}, {MAX_SEQUENCE_LENGTH}]."
        )
    if attention_masks.dtype not in (torch.bool, torch.int64):
        raise DistillationContractError(
            "attention_masks must be bool or int64."
        )
    if not bool(((attention_masks == 0) | (attention_masks == 1)).all()):
        raise DistillationContractError("attention_masks must contain only 0/1.")


def load_distill_prompt_cache(path: str | Path) -> DistillPromptFeatures:
    cache_path = Path(path).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(cache, dict) or cache.get("format_version") != 2:
        raise DistillationContractError(
            "Distillation prompt cache must be a format_version=2 dictionary."
        )
    required = {
        "prompt_ids",
        "source_sample_ids",
        "prompt_embeds",
        "attention_masks",
        "empty_prompt_embeds",
        "empty_prompt_attention_mask",
        "prompt_bank_fingerprint",
        "text_encoder_model",
        "max_sequence_length",
    }
    missing = sorted(required - set(cache))
    if missing:
        raise DistillationContractError(
            f"Distillation prompt cache is missing keys: {missing}"
        )
    prompt_ids = tuple(cache["prompt_ids"])
    source_ids = tuple(cache["source_sample_ids"])
    validate_prompt_feature_tensors(
        prompt_ids,
        source_ids,
        cache["prompt_embeds"],
        cache["attention_masks"],
    )
    empty_embeds = cache["empty_prompt_embeds"]
    empty_mask = cache["empty_prompt_attention_mask"]
    if empty_embeds.shape != (1, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
        raise DistillationContractError("Invalid empty prompt embedding shape.")
    if empty_embeds.dtype != torch.float16 or not bool(
        torch.isfinite(empty_embeds).all()
    ):
        raise DistillationContractError(
            "Empty prompt embeddings must be finite float16."
        )
    if empty_mask.shape != (1, MAX_SEQUENCE_LENGTH) or empty_mask.dtype not in (
        torch.bool,
        torch.int64,
    ):
        raise DistillationContractError("Invalid empty prompt attention mask.")
    if cache["max_sequence_length"] != MAX_SEQUENCE_LENGTH:
        raise DistillationContractError("Invalid max_sequence_length.")
    return DistillPromptFeatures(
        path=cache_path,
        prompt_ids=prompt_ids,
        source_sample_ids=source_ids,
        prompt_embeds=cache["prompt_embeds"].contiguous(),
        attention_masks=cache["attention_masks"].contiguous(),
        empty_prompt_embeds=empty_embeds.contiguous(),
        empty_prompt_attention_mask=empty_mask.contiguous(),
        prompt_bank_fingerprint=cache["prompt_bank_fingerprint"],
        text_encoder_model=cache["text_encoder_model"],
    )


def batched(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive.")
    for start in range(0, len(values), size):
        yield values[start : start + size]

