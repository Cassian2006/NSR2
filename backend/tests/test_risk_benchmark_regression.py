from __future__ import annotations

from scripts.benchmark_planners import evaluate_regression, summarize


def test_summarize_includes_risk_and_stability_fields() -> None:
    rows = [
        {
            "mode": "static",
            "planner": "astar",
            "status": "ok",
            "runtime_ms": 100.0,
            "distance_km": 200.0,
            "caution_len_km": 20.0,
            "risk_exposure": 5.0,
            "route_cost_effective_km": 205.0,
            "corridor_alignment": 0.5,
            "route_signature": "abc",
        },
        {
            "mode": "static",
            "planner": "astar",
            "status": "ok",
            "runtime_ms": 110.0,
            "distance_km": 200.0,
            "caution_len_km": 25.0,
            "risk_exposure": 6.0,
            "route_cost_effective_km": 206.0,
            "corridor_alignment": 0.6,
            "route_signature": "abc",
        },
    ]
    out = summarize(rows)
    grp = out["static:astar"]
    assert "avg_risk_exposure" in grp
    assert "stability_consistency" in grp
    assert grp["route_signature_unique"] == 1
    assert float(grp["stability_consistency"]) == 1.0


def test_evaluate_regression_detects_failures() -> None:
    baseline = {
        "static:astar": {
            "avg_runtime_ms": 100.0,
            "avg_distance_km": 200.0,
            "avg_risk_exposure": 2.0,
            "success_rate": 1.0,
            "stability_consistency": 1.0,
        }
    }
    current = {
        "static:astar": {
            "avg_runtime_ms": 160.0,
            "avg_distance_km": 240.0,
            "avg_risk_exposure": 3.0,
            "success_rate": 0.7,
            "stability_consistency": 0.7,
        }
    }
    report = evaluate_regression(
        current_summary=current,
        baseline_summary=baseline,
        runtime_max_increase_pct=0.35,
        distance_max_increase_pct=0.10,
        risk_max_increase_pct=0.20,
        success_rate_max_drop=0.05,
        stability_consistency_max_drop=0.1,
    )
    assert report["status"] == "fail"
    assert report["compared"]
    assert report["compared"][0]["status"] == "fail"


def test_evaluate_regression_pass_when_within_threshold() -> None:
    baseline = {
        "dynamic:dstar_lite": {
            "avg_runtime_ms": 120.0,
            "avg_distance_km": 210.0,
            "avg_risk_exposure": 2.5,
            "success_rate": 1.0,
            "stability_consistency": 1.0,
        }
    }
    current = {
        "dynamic:dstar_lite": {
            "avg_runtime_ms": 125.0,
            "avg_distance_km": 212.0,
            "avg_risk_exposure": 2.6,
            "success_rate": 0.98,
            "stability_consistency": 0.95,
        }
    }
    report = evaluate_regression(current_summary=current, baseline_summary=baseline)
    assert report["status"] == "pass"
