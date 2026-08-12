#!/usr/bin/env python3
"""Compatibility entry point for distilled-model evaluation.

Transformers 4.x returned a Tensor from ``CLIPModel.get_*_features`` while
Transformers 5.x returns ``BaseModelOutputWithPooling``.  The actual projected
feature Tensor lives in ``pooler_output`` in the newer API.  Keep the evaluator
logic versioned and patch its normalization boundary so both APIs work.
"""

from __future__ import annotations

from typing import Any

import torch

from scripts.distillation import evaluate_distilled_impl_v1 as _impl
from scripts.distillation.evaluate_distilled_impl_v1 import *  # noqa: F401,F403


def clip_feature_tensor(output: Any) -> torch.Tensor:
    """Return the projected rank-2 CLIP feature Tensor from either API."""
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    if isinstance(output, (tuple, list)):
        for value in reversed(output):
            if isinstance(value, torch.Tensor) and value.ndim == 2:
                return value
    raise TypeError(
        "CLIP feature output contains no rank-2 Tensor or pooler_output: "
        f"{type(output).__name__}"
    )


def F_normalize(output: Any) -> torch.Tensor:
    """Normalize CLIP features in FP32 after unwrapping model output."""
    tensor = clip_feature_tensor(output)
    return torch.nn.functional.normalize(tensor.float(), dim=-1)


# Functions imported above retain the globals of the versioned implementation.
# Patch that module's normalization boundary before ``evaluate`` is called.
_impl.F_normalize = F_normalize

