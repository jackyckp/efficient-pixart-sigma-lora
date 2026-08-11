"""Shared utilities for PixArt-Sigma progressive LoRA distillation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch


TRANSFORMER_MODEL = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
COMPONENT_MODEL = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"


def model_snapshot_source(model_id: str) -> str | Path:
    """Prefer a complete local Hugging Face snapshot, then fall back to Hub ID."""

    slug = "models--" + model_id.replace("/", "--")
    project_root = Path(__file__).resolve().parent.parent
    cache_roots = [project_root / ".cache" / "huggingface" / "hub"]
    if hf_home := os.environ.get("HF_HOME"):
        cache_roots.insert(0, Path(hf_home).expanduser().resolve() / "hub")
    for cache_root in cache_roots:
        model_root = cache_root / slug
        ref = model_root / "refs" / "main"
        candidates: list[Path] = []
        if ref.is_file():
            revision = ref.read_text(encoding="utf-8").strip()
            if revision:
                candidates.append(model_root / "snapshots" / revision)
        snapshots = model_root / "snapshots"
        if snapshots.is_dir():
            candidates.extend(path for path in snapshots.iterdir() if path.is_dir())
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
    return model_id


def resolve_adapter(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    direct = candidate / "adapter_config.json"
    nested = candidate / "lora_adapter" / "adapter_config.json"
    if direct.is_file():
        adapter = candidate
    elif nested.is_file():
        adapter = candidate / "lora_adapter"
    else:
        raise FileNotFoundError(
            f"No adapter_config.json found directly or under lora_adapter: {candidate}"
        )
    weights = adapter / "adapter_model.safetensors"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty adapter weights: {weights}")
    return adapter


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_caches(latent_path: Path, text_path: Path) -> dict:
    latent_path = latent_path.expanduser().resolve()
    text_path = text_path.expanduser().resolve()
    if not latent_path.is_file():
        raise FileNotFoundError(f"Missing latent cache: {latent_path}")
    if not text_path.is_file():
        raise FileNotFoundError(f"Missing text cache: {text_path}")

    latent_cache = torch.load(latent_path, map_location="cpu", weights_only=True)
    text_cache = torch.load(text_path, map_location="cpu", weights_only=True)
    required_latent = {"latents", "sample_ids", "manifest_fingerprint"}
    required_text = {
        "prompt_embeds",
        "attention_masks",
        "empty_prompt_embeds",
        "empty_prompt_attention_mask",
        "sample_ids",
        "manifest_fingerprint",
    }
    if missing := required_latent.difference(latent_cache):
        raise KeyError(f"Latent cache is missing keys: {sorted(missing)}")
    if missing := required_text.difference(text_cache):
        raise KeyError(f"Text cache is missing keys: {sorted(missing)}")
    if latent_cache["manifest_fingerprint"] != text_cache["manifest_fingerprint"]:
        raise ValueError("Latent and text cache fingerprints do not match.")
    if latent_cache["sample_ids"] != text_cache["sample_ids"]:
        raise ValueError("Latent and text cache sample order does not match.")

    latents = latent_cache["latents"]
    prompt_embeds = text_cache["prompt_embeds"]
    attention_masks = text_cache["attention_masks"]
    count = len(latent_cache["sample_ids"])
    expected = {
        "latents": (count, 4, 64, 64),
        "prompt_embeds": (count, 300, 4096),
        "attention_masks": (count, 300),
    }
    actual = {
        "latents": tuple(latents.shape),
        "prompt_embeds": tuple(prompt_embeds.shape),
        "attention_masks": tuple(attention_masks.shape),
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name} shape is {actual[name]}, expected {shape}")
    if latents.dtype != torch.float16 or prompt_embeds.dtype != torch.float16:
        raise TypeError("Latents and prompt embeddings must both be float16.")
    if attention_masks.dtype != torch.int64:
        raise TypeError("Prompt attention masks must be int64.")

    return {
        "latent_path": latent_path,
        "text_path": text_path,
        "latent_cache": latent_cache,
        "text_cache": text_cache,
        "latents": latents.contiguous(),
        "prompt_embeds": prompt_embeds.contiguous(),
        "attention_masks": attention_masks.contiguous(),
        "empty_prompt_embeds": text_cache["empty_prompt_embeds"].contiguous(),
        "empty_prompt_attention_mask": text_cache[
            "empty_prompt_attention_mask"
        ].contiguous(),
        "sample_ids": list(latent_cache["sample_ids"]),
        "fingerprint": latent_cache["manifest_fingerprint"],
    }


def make_ddim_scheduler(num_steps: int, device: torch.device | str = "cpu"):
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler.from_pretrained(
        model_snapshot_source(COMPONENT_MODEL),
        subfolder="scheduler",
        timestep_spacing="trailing",
        clip_sample=False,
    )
    scheduler.set_timesteps(num_steps, device=device)
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError(
            "This implementation expects epsilon prediction, got "
            f"{scheduler.config.prediction_type!r}."
        )
    return scheduler


def progressive_transition_triples(source_steps: int, student_steps: int) -> list:
    if source_steps != 2 * student_steps:
        raise ValueError(
            "Progressive halving requires source_steps == 2 * student_steps, "
            f"got {source_steps} -> {student_steps}."
        )
    source = make_ddim_scheduler(source_steps).timesteps.cpu().tolist()
    student = make_ddim_scheduler(student_steps).timesteps.cpu().tolist()
    if student != source[::2]:
        raise RuntimeError(
            "Trailing DDIM schedules are not nested as expected: "
            f"source[::2]={source[::2]}, student={student}"
        )
    triples = []
    for index, timestep in enumerate(student):
        source_index = 2 * index
        middle = source[source_index + 1]
        destination = (
            source[source_index + 2] if source_index + 2 < len(source) else -1
        )
        triples.append((int(timestep), int(middle), int(destination)))
    return triples


def load_transformer_with_adapter(
    adapter_path: Path,
    *,
    trainable: bool,
    merge_for_inference: bool,
):
    from diffusers import PixArtTransformer2DModel
    from peft import PeftModel

    adapter = resolve_adapter(adapter_path)
    base = PixArtTransformer2DModel.from_pretrained(
        model_snapshot_source(TRANSFORMER_MODEL),
        subfolder="transformer",
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base,
        adapter,
        is_trainable=trainable,
        low_cpu_mem_usage=True,
    )
    if merge_for_inference:
        if trainable:
            raise ValueError("A trainable adapter cannot be merged for inference.")
        model = model.eval().merge_and_unload(safe_merge=True).eval()
    elif trainable:
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if "lora_" not in name:
                    raise RuntimeError(f"Unexpected trainable parameter: {name}")
                parameter.data = parameter.data.float()
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
    return model, adapter


def model_epsilon(
    model,
    latents: torch.Tensor,
    timestep: int | torch.Tensor,
    prompt_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    guidance_scale: float = 1.0,
    empty_prompt_embeds: torch.Tensor | None = None,
    empty_attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch = latents.shape[0]
    device = latents.device
    if isinstance(timestep, int):
        timestep_tensor = torch.full(
            (batch,), timestep, device=device, dtype=torch.long
        )
    else:
        timestep_tensor = timestep.to(device=device, dtype=torch.long).reshape(-1)
        if timestep_tensor.numel() == 1 and batch > 1:
            timestep_tensor = timestep_tensor.expand(batch)

    if guidance_scale > 1.0:
        if empty_prompt_embeds is None or empty_attention_mask is None:
            raise ValueError("CFG requires empty prompt embeddings and mask.")
        empty_prompt_embeds = empty_prompt_embeds.expand(batch, -1, -1)
        empty_attention_mask = empty_attention_mask.expand(batch, -1)
        latent_input = torch.cat((latents, latents), dim=0)
        encoder_states = torch.cat((empty_prompt_embeds, prompt_embeds), dim=0)
        encoder_mask = torch.cat((empty_attention_mask, attention_mask), dim=0)
        timestep_input = torch.cat((timestep_tensor, timestep_tensor), dim=0)
    else:
        latent_input = latents
        encoder_states = prompt_embeds
        encoder_mask = attention_mask
        timestep_input = timestep_tensor

    output = model(
        latent_input,
        encoder_hidden_states=encoder_states,
        encoder_attention_mask=encoder_mask,
        timestep=timestep_input,
        added_cond_kwargs={"resolution": None, "aspect_ratio": None},
    ).sample
    epsilon = output.chunk(2, dim=1)[0]
    if guidance_scale > 1.0:
        epsilon_uncond, epsilon_cond = epsilon.chunk(2, dim=0)
        epsilon = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)
    return epsilon


def derive_epsilon_for_endpoint(
    scheduler,
    sample_at_t: torch.Tensor,
    endpoint: torch.Tensor,
    timestep: int,
    destination_timestep: int,
) -> torch.Tensor:
    """Solve for epsilon so one deterministic DDIM step reaches endpoint."""

    device = sample_at_t.device
    dtype = torch.float32
    alpha_t_bar = scheduler.alphas_cumprod[timestep].to(device=device, dtype=dtype)
    if destination_timestep >= 0:
        alpha_s_bar = scheduler.alphas_cumprod[destination_timestep].to(
            device=device, dtype=dtype
        )
    else:
        alpha_s_bar = scheduler.final_alpha_cumprod.to(device=device, dtype=dtype)
    alpha_t = alpha_t_bar.sqrt()
    sigma_t = (1.0 - alpha_t_bar).sqrt()
    alpha_s = alpha_s_bar.sqrt()
    sigma_s = (1.0 - alpha_s_bar).sqrt()

    x_t = sample_at_t.float()
    x_s = endpoint.float()
    ratio = sigma_s / sigma_t
    denominator = alpha_s - ratio * alpha_t
    if denominator.abs().item() < 1e-8:
        raise FloatingPointError("Degenerate DDIM endpoint conversion denominator.")
    predicted_x0 = (x_s - ratio * x_t) / denominator
    epsilon = (x_t - alpha_t * predicted_x0) / sigma_t
    return epsilon.to(dtype=sample_at_t.dtype)
