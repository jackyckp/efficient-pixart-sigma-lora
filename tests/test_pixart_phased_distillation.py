from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.distillation.common import (
    EXPECTED_ADAPTER_PARAMETERS,
    EXPECTED_ADAPTER_TENSORS,
    TEACHER_TIMESTEPS,
    build_prompt_records,
    deterministic_jump,
    deterministic_trajectory_seed,
    effective_epsilon_target,
    inspect_adapter,
    load_prompt_bank,
    phase_pairs,
    pseudo_huber_loss,
    save_prompt_bank,
    split_epsilon_prediction,
    state_timestep,
)
from scripts.distillation.evaluate_distilled import (
    expected_image_records,
    unbiased_cmmd,
)
from scripts.distillation.train_phased_distill_lora import apply_stage_defaults


def test_phase_pairs_map_exact_teacher_trajectory_states() -> None:
    assert len(TEACHER_TIMESTEPS) == 20
    assert phase_pairs(4) == ((0, 5), (5, 10), (10, 15), (15, 20))
    assert phase_pairs(2) == ((0, 10), (10, 20))
    assert [state_timestep(index) for index in (0, 5, 10, 15, 20)] == [
        999,
        749,
        500,
        250,
        -1,
    ]
    with pytest.raises(ValueError):
        phase_pairs(3)


@pytest.mark.parametrize(
    ("start_timestep", "target_timestep"),
    ((999, 749), (749, 500), (500, 250), (250, -1), (999, 500), (500, -1)),
)
def test_effective_epsilon_round_trips_arbitrary_jump(
    start_timestep: int,
    target_timestep: int,
) -> None:
    generator = torch.Generator().manual_seed(7)
    sample = torch.randn((2, 4, 8, 8), generator=generator)
    target = torch.randn((2, 4, 8, 8), generator=generator)
    alphas = torch.linspace(0.9999, 0.001, 1000)
    epsilon = effective_epsilon_target(
        sample,
        target,
        start_timestep,
        target_timestep,
        alphas,
    )
    reconstructed = deterministic_jump(
        sample,
        epsilon,
        start_timestep,
        target_timestep,
        alphas,
    )
    torch.testing.assert_close(reconstructed, target, rtol=1e-4, atol=1e-4)


def test_jump_rejects_invalid_direction_and_shape() -> None:
    alphas = torch.linspace(0.9999, 0.001, 1000)
    sample = torch.zeros((1, 4, 8, 8))
    with pytest.raises(ValueError, match="earlier"):
        deterministic_jump(sample, sample, 500, 500, alphas)
    with pytest.raises(ValueError, match="shapes"):
        deterministic_jump(sample, sample[:, :3], 999, 500, alphas)


def test_learned_sigma_split_and_pseudo_huber() -> None:
    output = torch.cat(
        [torch.ones((1, 4, 2, 2)), torch.full((1, 4, 2, 2), 9.0)], dim=1
    )
    epsilon = split_epsilon_prediction(output)
    assert epsilon.shape == (1, 4, 2, 2)
    assert torch.equal(epsilon, torch.ones_like(epsilon))
    assert pseudo_huber_loss(epsilon, epsilon).item() == pytest.approx(0.0)
    with pytest.raises(ValueError):
        pseudo_huber_loss(epsilon, epsilon, c=0)
    with pytest.raises(ValueError):
        split_epsilon_prediction(torch.zeros((1, 4, 2, 2)))


def test_trajectory_seed_is_deterministic_and_replica_specific() -> None:
    first = deterministic_trajectory_seed("plant/220::original", 0)
    assert first == deterministic_trajectory_seed("plant/220::original", 0)
    assert first != deterministic_trajectory_seed("plant/220::original", 1)
    assert 0 <= first < 2**31


def test_prompt_bank_builds_three_unique_variants_per_plant(tmp_path: Path) -> None:
    manifest = [
        {
            "sample_id": f"plant/{index:03d}",
            "category": "plant",
            "caption": (
                f"A pine tree number {index}. A quiet background, "
                "Chinese ink wash painting style, Sumi-e"
            ),
        }
        for index in range(209)
    ]
    records = build_prompt_records(manifest)
    assert len(records) == 627
    assert {record["variant"] for record in records} == {
        "original",
        "subject",
        "styled",
    }
    assert len({record["prompt_id"] for record in records}) == 627
    assert sum("shuimo hua" in record["prompt"] for record in records) == 418
    path = tmp_path / "prompt_bank.jsonl"
    fingerprint = save_prompt_bank(path, records)
    loaded = load_prompt_bank(path)
    assert loaded.records == records
    assert loaded.fingerprint == fingerprint


def test_evaluation_manifest_expands_prompt_seed_cartesian_product() -> None:
    prompts = (
        {"prompt_id": "eval-01", "prompt": "pine", "seeds": [1, 2, 3, 4]},
        {"prompt_id": "eval-02", "prompt": "lotus", "seeds": [5, 6, 7, 8]},
    )
    records = expected_image_records(prompts)
    assert len(records) == 8
    assert records[0]["filename"] == "eval-01_seed1.png"
    assert records[-1]["filename"] == "eval-02_seed8.png"


def test_unbiased_cmmd_is_finite_and_symmetric() -> None:
    generator = torch.Generator().manual_seed(11)
    left = torch.nn.functional.normalize(
        torch.randn((8, 16), generator=generator), dim=1
    )
    right = torch.nn.functional.normalize(
        torch.randn((9, 16), generator=generator), dim=1
    )
    forward = unbiased_cmmd(left, right)
    reverse = unbiased_cmmd(right, left)
    assert forward == pytest.approx(reverse, abs=1e-7)
    assert torch.isfinite(torch.tensor(forward))


def test_training_stage_defaults_are_locked() -> None:
    four = type("Args", (), {
        "target_steps": 4,
        "max_train_steps": None,
        "learning_rate": None,
        "checkpoint_every_steps": None,
    })()
    apply_stage_defaults(four)
    assert (four.max_train_steps, four.learning_rate, four.checkpoint_every_steps) == (
        2_000,
        5e-6,
        500,
    )
    two = type("Args", (), {
        "target_steps": 2,
        "max_train_steps": None,
        "learning_rate": None,
        "checkpoint_every_steps": None,
    })()
    apply_stage_defaults(two)
    assert (two.max_train_steps, two.learning_rate, two.checkpoint_every_steps) == (
        10_000,
        2e-6,
        1_000,
    )


@pytest.mark.parametrize(
    "relative",
    (
        "outputs/style_teacher/plant_n209_steps10200/r16_lr1e-05",
        "outputs/style_teacher/best_ink_wash_lora_plant209_step4000",
    ),
)
def test_local_teacher_adapter_contract_when_present(relative: str) -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / relative
    if not path.exists():
        pytest.skip("Local ignored teacher output is not available on this checkout.")
    inspection = inspect_adapter(path)
    assert inspection["rank"] == 16
    assert inspection["lora_alpha"] == 16
    assert inspection["tensor_count"] == EXPECTED_ADAPTER_TENSORS
    assert inspection["parameter_count"] == EXPECTED_ADAPTER_PARAMETERS
    assert len(inspection["adapter_sha256"]) == 64


def test_evaluation_summary_json_schema_is_serializable() -> None:
    payload = {
        "status": "PASS",
        "gates": {
            "clip_at_least_90_percent": True,
            "cmmd_at_most_1_5x_teacher": True,
            "median_latency_speedup_at_least_5x": True,
        },
    }
    assert json.loads(json.dumps(payload)) == payload
