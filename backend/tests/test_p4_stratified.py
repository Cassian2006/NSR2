from __future__ import annotations

from app.core.p4_stratified import build_stratified_eval_report


def _payload() -> dict:
    rows: list[dict] = []
    for i in range(4):
        rows.append(
            {
                "mode": "static",
                "timestamp": f"2024-07-01_0{i}",
                "planner": "astar",
                "repeat_idx": i,
                "status": "ok",
                "distance_km": 2000.0 + i * 10,
                "caution_len_km": 100.0 + i * 5,
                "route_cost_effective_km": 2100.0 + i * 10,
                "risk_exposure": 20.0 + i * 2,
            }
        )
    for i in range(4):
        rows.append(
            {
                "mode": "dynamic",
                "window_start": f"2024-07-02_0{i}",
                "window_end": f"2024-07-02_1{i}",
                "planner": "dstar_lite",
                "repeat_idx": i,
                "status": "ok",
                "replan_count": 2,
                "avg_replan_ms": 120.0 + i,
                "distance_km": 2600.0 + i * 20,
                "caution_len_km": 80.0 + i * 5,
                "route_cost_effective_km": 2700.0 + i * 15,
                "risk_exposure": 60.0 + i * 4,
            }
        )
    rows.append(
        {
            "mode": "dynamic",
            "window_start": "2024-07-03_00",
            "window_end": "2024-07-03_03",
            "planner": "astar",
            "repeat_idx": 0,
            "status": "fail",
            "replan_count": 1,
            "avg_replan_ms": 100.0,
            "distance_km": 2500.0,
            "caution_len_km": 120.0,
            "route_cost_effective_km": 2700.0,
            "risk_exposure": 55.0,
            "detail": "no feasible route",
        }
    )
    return {"rows": rows}


def test_stratified_eval_contains_failure_index_and_traceability() -> None:
    report = build_stratified_eval_report(benchmark_payload=_payload())
    assert report["report_version"] == "p4_stratified_eval_v1"
    assert report["summary"]["total_rows"] == 9
    failures = report["failure_case_index"]
    assert failures
    trace = failures[0]["trace"]
    assert "case_ref" in trace and trace["case_ref"]
    assert "planner" in trace and trace["planner"]


def test_stratified_eval_dimension_coverage_pass() -> None:
    report = build_stratified_eval_report(benchmark_payload=_payload())
    cover = report["summary"]["dimension_coverage"]
    assert cover["volatility_band"]["ok"] is True
    assert cover["risk_band"]["ok"] is True
    assert cover["distance_band"]["ok"] is True
    assert report["summary"]["status"] == "PASS"


def test_stratified_eval_warn_when_missing_dimension_values() -> None:
    payload = {
        "rows": [
            {
                "mode": "static",
                "timestamp": "2024-07-01_00",
                "planner": "astar",
                "repeat_idx": 0,
                "status": "ok",
                "distance_km": 1000.0,
                "caution_len_km": 10.0,
                "route_cost_effective_km": 1010.0,
                "risk_exposure": 5.0,
            }
        ]
    }
    report = build_stratified_eval_report(benchmark_payload=payload)
    assert report["summary"]["status"] == "WARN"
    cover = report["summary"]["dimension_coverage"]
    assert cover["volatility_band"]["ok"] is False

