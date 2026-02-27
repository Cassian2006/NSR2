from __future__ import annotations

from scripts.benchmark_planners import build_temporal_compare, summarize


def _rows() -> list[dict]:
    return [
        {
            "mode": "dynamic",
            "planner": "astar",
            "status": "ok",
            "runtime_ms": 1200.0,
            "distance_km": 1500.0,
            "caution_len_km": 100.0,
            "corridor_alignment": 0.25,
            "risk_exposure": 120.0,
            "route_cost_effective_km": 1650.0,
            "avg_replan_ms": 210.0,
            "route_safety": 0.933333,
            "route_signature": "sig-a-1",
        },
        {
            "mode": "dynamic",
            "planner": "astar",
            "status": "ok",
            "runtime_ms": 1210.0,
            "distance_km": 1510.0,
            "caution_len_km": 90.0,
            "corridor_alignment": 0.24,
            "risk_exposure": 125.0,
            "route_cost_effective_km": 1660.0,
            "avg_replan_ms": 200.0,
            "route_safety": 0.940397,
            "route_signature": "sig-a-1",
        },
        {
            "mode": "dynamic",
            "planner": "dstar_lite",
            "status": "ok",
            "runtime_ms": 900.0,
            "distance_km": 1495.0,
            "caution_len_km": 80.0,
            "corridor_alignment": 0.31,
            "risk_exposure": 110.0,
            "route_cost_effective_km": 1600.0,
            "avg_replan_ms": 120.0,
            "route_safety": 0.946488,
            "route_signature": "sig-d-1",
        },
        {
            "mode": "dynamic",
            "planner": "dstar_lite",
            "status": "ok",
            "runtime_ms": 890.0,
            "distance_km": 1492.0,
            "caution_len_km": 85.0,
            "corridor_alignment": 0.29,
            "risk_exposure": 112.0,
            "route_cost_effective_km": 1595.0,
            "avg_replan_ms": 118.0,
            "route_safety": 0.943030,
            "route_signature": "sig-d-1",
        },
        {
            "mode": "dynamic",
            "planner": "any_angle",
            "status": "fail",
            "runtime_ms": 1400.0,
            "distance_km": 0.0,
            "caution_len_km": 0.0,
            "corridor_alignment": 0.0,
            "risk_exposure": 0.0,
            "route_cost_effective_km": 0.0,
            "avg_replan_ms": 0.0,
            "route_safety": 0.0,
            "route_signature": "",
        },
    ]


def test_dynamic_benchmark_summary_contains_replan_latency() -> None:
    summary = summarize(_rows())
    assert "dynamic:astar" in summary
    assert "dynamic:dstar_lite" in summary
    assert "avg_replan_latency_ms" in summary["dynamic:astar"]
    assert "avg_replan_latency_ms" in summary["dynamic:dstar_lite"]
    assert summary["dynamic:dstar_lite"]["avg_replan_latency_ms"] < summary["dynamic:astar"]["avg_replan_latency_ms"]


def test_temporal_compare_contains_required_axes_and_ranking() -> None:
    summary = summarize(_rows())
    compare = build_temporal_compare(summary=summary, planners=["astar", "dstar_lite", "any_angle"])
    assert compare["benchmark_version"] == "temporal_benchmark_v1"
    assert compare["axes"] == ["completion_rate", "total_cost_km", "replan_latency_ms", "stability", "route_safety"]
    planners = compare["planners"]
    assert "astar" in planners
    assert "dstar_lite" in planners
    assert planners["astar"]["completion_rate"] == 1.0
    assert planners["dstar_lite"]["completion_rate"] == 1.0
    assert planners["dstar_lite"]["replan_latency_ms"] < planners["astar"]["replan_latency_ms"]
    assert 0.0 <= planners["dstar_lite"]["route_safety"] <= 1.0
    assert compare["ranking"]
    assert compare["ranking"][0]["planner"] in {"astar", "dstar_lite"}
