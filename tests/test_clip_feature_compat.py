from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.distillation.evaluate_distilled_impl import (
    F_normalize,
    clip_feature_tensor,
)


def test_clip_feature_tensor_accepts_legacy_tensor() -> None:
    tensor = torch.tensor([[3.0, 4.0]])
    assert clip_feature_tensor(tensor) is tensor
    assert torch.allclose(F_normalize(tensor), torch.tensor([[0.6, 0.8]]))


def test_clip_feature_tensor_unwraps_transformers_5_pooler_output() -> None:
    pooled = torch.tensor([[0.0, 5.0]])
    output = SimpleNamespace(pooler_output=pooled)
    assert clip_feature_tensor(output) is pooled
    assert torch.equal(F_normalize(output), torch.tensor([[0.0, 1.0]]))


def test_evaluator_uses_compatibility_normalizer() -> None:
    output = SimpleNamespace(pooler_output=torch.tensor([[2.0, 0.0]]))
    assert torch.equal(
        F_normalize(output),
        torch.tensor([[1.0, 0.0]]),
    )


def test_clip_feature_tensor_rejects_unknown_output() -> None:
    with pytest.raises(TypeError, match="pooler_output"):
        clip_feature_tensor(object())

