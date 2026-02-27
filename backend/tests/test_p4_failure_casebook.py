from __future__ import annotations

from app.core.p4_failure_casebook import build_failure_casebook, infer_root_cause_tags


def test_infer_root_cause_tags_threshold_boundary_uncertainty() -> None:
    row = {
        "detail": "risk budget threshold exceeded near boundary",
        "risk_budget_usage": 1.2,
        "start_adjusted": True,
        "uncertainty_penalty_mean": 0.35,
    }
    tags = infer_root_cause_tags(row, degraded=True, uncertainty_penalty_threshold=0.2)
    assert "threshold" in tags
    assert "boundary" in tags
    assert "model_uncertainty" in tags
    assert "planner_degradation" in tags


def test_build_failure_casebook_collects_failed_and_degraded_cases() -> None:
    payload = {
        "benchmark_version": "temporal_benchmark_v1",
        "created_at": "2026-02-15T00:00:00Z",
        "start": {"lat": 70.5, "lon": 30.0},
        "goal": {"lat": 72.0, "lon": 150.0},
        "rows": [
            {
                "mode": "static",
                "timestamp": "2024-07-01_00",
                "planner": "astar",
                "repeat_idx": 0,
                "status": "ok",
                "runtime_ms": 100.0,
                "distance_km": 1000.0,
                "route_cost_effective_km": 1000.0,
                "risk_exposure": 10.0,
            },
            {
                "mode": "static",
                "timestamp": "2024-07-01_00",
                "planner": "dstar_lite",
                "repeat_idx": 0,
                "status": "ok",
                "runtime_ms": 250.0,
                "distance_km": 1000.0,
                "route_cost_effective_km": 1200.0,
                "risk_exposure": 18.0,
            },
            {
                "mode": "dynamic",
                "window_start": "2024-07-01_00",
                "window_end": "2024-07-01_06",
                "planner": "astar",
                "repeat_idx": 0,
                "status": "fail",
                "runtime_ms": 0.0,
                "distance_km": 0.0,
                "route_cost_effective_km": 0.0,
                "risk_exposure": 0.0,
                "detail": "no data for timestamp",
            },
        ],
    }
    report = build_failure_casebook(
        benchmark_payload=payload,
        runtime_degrade_pct=0.5,
        cost_degrade_pct=0.15,
        risk_degrade_pct=0.2,
        uncertainty_penalty_threshold=0.2,
    )
    assert report["report_version"] == "p4_failure_casebook_v1"
    assert report["summary"]["failed_count"] >= 1
    assert report["summary"]["degraded_count"] >= 1
    assert report["summary"]["status"] == "FAIL"
    cases = report["cases"]
    assert any(c["case_type"] == "failed" for c in cases)
    assert any(c["case_type"] == "degraded" for c in cases)
    first = cases[0]
    assert "replay_hint" in first
    assert "scenario_ref" in first["replay_hint"]
    assert "policy_defaults" in first["replay_hint"]

