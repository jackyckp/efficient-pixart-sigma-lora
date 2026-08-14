from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.training.train_local_latent_lora import (
    AssetValidationError,
    deterministic_subset_ids,
    load_latent_bundle,
)
from scripts.training.train_style_teacher_sweep import (
    FIXED_GUIDANCE_SCALE,
    FIXED_INFERENCE_STEPS,
    build_run_matrix,
    main,
)


ROOT = Path(__file__).resolve().parents[1]
LATENT_BUNDLE = ROOT / "data" / "archives" / "clean_latents_512.zip"


def test_plant_category_selection_is_strict_and_complete() -> None:
    if not LATENT_BUNDLE.is_file():
        pytest.skip(f"Latent bundle not present: {LATENT_BUNDLE}")
    latent_bundle = load_latent_bundle(LATENT_BUNDLE)
    plant_ids = deterministic_subset_ids(
        latent_bundle.sample_ids,
        209,
        42,
        "plant",
    )
    assert len(plant_ids) == 209
    assert all(sample_id.startswith("plant/") for sample_id in plant_ids)
    assert plant_ids == deterministic_subset_ids(
        latent_bundle.sample_ids,
        209,
        42,
        "plant",
    )
    with pytest.raises(AssetValidationError, match=r"\[1, 209\]"):
        deterministic_subset_ids(
            latent_bundle.sample_ids,
            255,
            42,
            "plant",
        )


def test_teacher_matrix_has_four_ranks_at_one_learning_rate() -> None:
    matrix = build_run_matrix((4, 8, 16, 32), 1e-5)
    assert [(run["rank"], run["learning_rate"]) for run in matrix] == [
        (4, 1e-5),
        (8, 1e-5),
        (16, 1e-5),
        (32, 1e-5),
    ]
    assert len({run["name"] for run in matrix}) == 4


def test_teacher_sweep_dry_run_writes_fixed_inference_plan(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "teacher"
    result = main(
        [
            "--ranks",
            "4",
            "8",
            "16",
            "32",
            "--learning-rate",
            "1e-5",
            "--num-images",
            "260",
            "--max-train-steps",
            "10000",
            "--checkpoint-every-steps",
            "1000",
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )
    assert result == 0
    plan = json.loads(
        (output_root / "sweep_plan.json").read_text(encoding="utf-8")
    )
    assert plan["dataset_scope"] == "all_categories"
    assert plan["category_counts"] == {
        "animal": 30,
        "others": 10,
        "plant": 209,
        "web": 11,
    }
    assert plan["num_images"] == 260
    assert plan["fixed_inference_steps"] == FIXED_INFERENCE_STEPS == 20
    assert plan["fixed_guidance_scale"] == FIXED_GUIDANCE_SCALE == 1.5
    assert plan["max_train_steps_per_model"] == 10_000
    assert plan["checkpoint_every_steps"] == 1_000
    assert plan["learning_rate"] == 1e-5
    assert len(plan["runs"]) == 4
    assert 7.7 < plan["estimated_total_hours"] < 8.0
    for run in plan["runs"]:
        command = run["command"]
        assert "--category" not in command
        assert "--num-images" in command
        assert "260" in command
        assert "--inference-steps" in command
        assert "20" in command
        assert "--guidance-scale" in command
        assert "1.5" in command
        assert "--checkpoint-every-steps" in command
        assert "1000" in command
        assert "--run-role" in command
        assert "style_teacher" in command
