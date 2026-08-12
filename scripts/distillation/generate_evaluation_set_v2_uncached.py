"""Fair-latency evaluation layer that avoids per-step CPU trajectory copies."""

from __future__ import annotations

import torch

from scripts.distillation import generate_evaluation_set_impl as _impl
from scripts.distillation.common import (
    LATENT_SHAPE,
    TEACHER_TIMESTEPS,
    split_epsilon_prediction,
)
from scripts.distillation.generate_evaluation_set_impl import *  # noqa: F401,F403


def _generate_teacher_final(
    *,
    transformer,
    scheduler,
    prompt_embed,
    prompt_mask,
    empty_embed,
    empty_mask,
    seed,
    guidance_scale,
    device,
):
    scheduler.set_timesteps(20, device=device)
    if tuple(int(value) for value in scheduler.timesteps.tolist()) != TEACHER_TIMESTEPS:
        raise RuntimeError("Teacher evaluation timestep schedule changed.")
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(
        (1, *LATENT_SHAPE),
        generator=generator,
        device=device,
        dtype=torch.float16,
    ) * scheduler.init_noise_sigma
    use_cfg = guidance_scale > 1.0
    embeds = torch.cat([empty_embed, prompt_embed]) if use_cfg else prompt_embed
    masks = torch.cat([empty_mask, prompt_mask]) if use_cfg else prompt_mask
    with torch.inference_mode():
        for timestep in scheduler.timesteps:
            model_input = torch.cat([latent, latent]) if use_cfg else latent
            model_input = scheduler.scale_model_input(model_input, timestep)
            output = transformer(
                model_input,
                encoder_hidden_states=embeds,
                encoder_attention_mask=masks,
                timestep=timestep.reshape(1).expand(model_input.shape[0]),
                added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                return_dict=False,
            )[0]
            if use_cfg:
                unconditional, conditional = output.chunk(2)
                output = unconditional + guidance_scale * (
                    conditional - unconditional
                )
            latent = scheduler.step(
                split_epsilon_prediction(output),
                timestep,
                latent,
                return_dict=False,
            )[0]
    if not bool(torch.isfinite(latent).all()):
        raise FloatingPointError("Teacher evaluation latent is not finite.")
    return latent.to("cpu", dtype=torch.float16)


_impl.generate_teacher_trajectory = _generate_teacher_final
generate_sets = _impl.generate_sets

