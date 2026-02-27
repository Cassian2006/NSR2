from __future__ import annotations

from app.core.p4_metrics import (
    build_benchmark_metric_bundle,
    compute_route_safety,
    metric_dictionary_payload,
    validate_metric_completeness,
)


def _sample_summary() -> dict:
    return {
        "static:dstar_lite": {"avg_effective_cost_km": 1200.0, "avg_route_safety": 0.95},
        "dynamic:dstar_lite": {"avg_effective_cost_km": 1100.0, "avg_route_safety": 0.93},
    }


def _sample_temporal_compare() -> dict:
    return {
        "planners": {
            "dstar_lite": {
                "completion_rate": 1.0,
                "total_cost_km": 1100.0,
                "replan_latency_ms": 120.0,
                "stability": 0.9,
                "risk_exposure": 10.0,
                "route_safety": 0.93,
            }
        }
    }


def _sample_rows() -> list[dict]:
    return [
        {
            "status": "ok",
            "distance_km": 1000.0,
            "caution_len_km": 80.0,
            "risk_exposure": 10.0,
            "route_cost_effective_km": 1100.0,
            "corridor_alignment": 0.3,
            "risk_constraint_satisfied": True,
            "risk_budget_usage": 0.7,
        },
        {
            "status": "ok",
            "distance_km": 900.0,
            "caution_len_km": 90.0,
            "risk_exposure": 11.0,
            "route_cost_effective_km": 1110.0,
            "corridor_alignment": 0.25,
            "risk_constraint_satisfied": False,
            "risk_budget_usage": 1.1,
        },
    ]


def test_metric_dictionary_contains_core_and_derived() -> None:
    payload = metric_dictionary_payload()
    core_ids = {x["id"] for x in payload["core_metrics"]}
    derived_ids = {x["id"] for x in payload["derived_metrics"]}
    assert {"completion_rate", "total_cost_km", "replan_latency_ms", "stability", "risk_exposure", "route_safety"} <= core_ids
    assert {"dynamic_vs_static_gain_pct", "risk_constraint_violation_rate", "explain_consistency"} <= derived_ids


def test_validate_metric_completeness_pass() -> None:
    report = validate_metric_completeness(
        summary=_sample_summary(),
        temporal_compare=_sample_temporal_compare(),
        rows=_sample_rows(),
    )
    assert report["status"] == "PASS"


def test_validate_metric_completeness_warn_when_violation_rate_unavailable() -> None:
    rows = [
        {
            "status": "ok",
            "distance_km": 1000.0,
            "caution_len_km": 100.0,
            "risk_exposure": 10.0,
            "route_cost_effective_km": 1100.0,
            "corridor_alignment": 0.2,
        }
    ]
    report = validate_metric_completeness(
        summary=_sample_summary(),
        temporal_compare=_sample_temporal_compare(),
        rows=rows,
    )
    assert report["status"] == "WARN"


def test_validate_metric_completeness_fail_when_core_metric_missing() -> None:
    temporal_compare = {"planners": {"dstar_lite": {"completion_rate": 1.0}}}
    report = validate_metric_completeness(
        summary={},
        temporal_compare=temporal_compare,
        rows=_sample_rows(),
    )
    assert report["status"] == "FAIL"


def test_build_benchmark_metric_bundle_contains_validation() -> None:
    out = build_benchmark_metric_bundle(
        summary=_sample_summary(),
        temporal_compare=_sample_temporal_compare(),
        rows=_sample_rows(),
    )
    assert out["dictionary_version"] == "p4_metric_dictionary_v1"
    assert out["validation"]["status"] == "PASS"


def test_compute_route_safety_bounds() -> None:
    assert 0.0 <= compute_route_safety(100.0, 20.0) <= 1.0
    assert compute_route_safety(100.0, 0.0) == 1.0
    assert compute_route_safety(100.0, 200.0) == 0.0

