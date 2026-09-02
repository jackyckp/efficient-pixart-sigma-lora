from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "evaluation" / "style_teacher_vs_official_base_30prompt.json"


def load_result() -> dict[str, object]:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_stage1_record_uses_matched_base_and_teacher_protocol() -> None:
    result = load_result()
    assert result["status"] == "PASS"
    protocol = result["protocol"]
    assert protocol["num_prompts"] == 30
    assert sum(protocol["category_counts"].values()) == 30
    assert protocol["seed"] == 42
    assert protocol["num_inference_steps"] == 20
    assert protocol["guidance_scale"] == 1.5
    assert protocol["clip_model"] == "openai/clip-vit-base-patch32"

    rows = result["results"]
    assert [row["model_id"] for row in rows] == [
        "official_base",
        "style_teacher_plant209_step4000",
    ]
    assert rows[0]["adapter"] is None
    assert rows[1]["adapter"]["rank"] == 16


def test_stage1_reported_deltas_match_recorded_metrics() -> None:
    result = load_result()
    base, teacher = result["results"]
    comparison = result["comparison"]

    clip_gain = teacher["mean_clip_score"] - base["mean_clip_score"]
    clip_gain_percent = clip_gain / base["mean_clip_score"] * 100
    cmmd_reduction = base["cmmd_to_reference"] - teacher["cmmd_to_reference"]
    cmmd_reduction_percent = cmmd_reduction / base["cmmd_to_reference"] * 100

    assert math.isclose(
        comparison["absolute_clip_gain"], clip_gain, abs_tol=1e-12
    )
    assert math.isclose(
        comparison["relative_clip_gain_percent"],
        clip_gain_percent,
        abs_tol=5e-5,
    )
    assert math.isclose(
        comparison["absolute_cmmd_reduction"],
        cmmd_reduction,
        abs_tol=1e-12,
    )
    assert math.isclose(
        comparison["relative_cmmd_reduction_percent"],
        cmmd_reduction_percent,
        abs_tol=5e-5,
    )


def test_readme_links_stage1_record_and_warns_against_cross_stage_comparison() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Stage 1 result: official base -> 20-step style teacher" in readme
    assert "style_teacher_vs_official_base_30prompt.json" in readme
    assert "should not be compared across the two stages" in readme
