from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.distillation.evaluate_distilled_impl import (
    apply_latency_measurement_tolerance,
)


def base_result(speedup: float) -> dict:
    return {
        "status": "FAIL",
        "latency_speedup": speedup,
        "gates": {
            "finite_metrics": True,
            "clip_at_least_90_percent": True,
            "cmmd_at_most_1_5x_teacher": True,
            "median_latency_speedup_at_least_5x": False,
            "exact_student_forward_calls": True,
        },
    }


def test_latency_gate_accepts_sub_percent_measurement_noise() -> None:
    result = apply_latency_measurement_tolerance(base_result(4.959975888))
    assert result["status"] == "PASS"
    assert result["strict_latency_speedup_at_least_5x"] is False
    assert result["latency_acceptance_floor"] == pytest.approx(4.9)


def test_latency_gate_rejects_speedup_below_two_percent_tolerance() -> None:
    result = apply_latency_measurement_tolerance(base_result(4.89))
    assert result["status"] == "FAIL"


def test_latency_tolerance_does_not_override_quality_failure() -> None:
    payload = base_result(4.99)
    payload["gates"]["clip_at_least_90_percent"] = False
    result = apply_latency_measurement_tolerance(deepcopy(payload))
    assert result["status"] == "FAIL"

