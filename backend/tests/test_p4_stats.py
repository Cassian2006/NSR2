from __future__ import annotations

from app.core.p4_stats import bootstrap_mean_ci, build_significance_report, permutation_test_mean_diff


def test_bootstrap_mean_ci_returns_valid_interval() -> None:
    out = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], n_boot=400, alpha=0.05, seed=7)
    assert out["n"] == 4
    assert out["mean"] is not None
    assert out["ci_low"] is not None and out["ci_high"] is not None
    assert float(out["ci_low"]) <= float(out["mean"]) <= float(out["ci_high"])


def test_permutation_detects_shifted_means() -> None:
    a = [10.0, 11.0, 12.0, 13.0, 14.0]
    b = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = permutation_test_mean_diff(a, b, n_perm=2000, seed=42)
    assert out["mean_diff"] is not None and float(out["mean_diff"]) > 0.0
    assert out["p_value"] is not None and float(out["p_value"]) < 0.05


def test_build_significance_report_contains_confidence_and_conclusions() -> None:
    rows: list[dict] = []
    for i in range(8):
        rows.append(
            {
                "mode": "dynamic",
                "planner": "astar",
                "status": "ok",
                "distance_km": 1000.0 + i,
                "caution_len_km": 150.0 + i,
                "route_cost_effective_km": 1300.0 + i,
                "avg_replan_ms": 350.0 + i,
                "risk_exposure": 120.0 + i,
            }
        )
        rows.append(
            {
                "mode": "dynamic",
                "planner": "dstar_lite",
                "status": "ok",
                "distance_km": 1000.0 + i,
                "caution_len_km": 30.0 + i,
                "route_cost_effective_km": 1120.0 + i,
                "avg_replan_ms": 120.0 + i,
                "risk_exposure": 50.0 + i,
            }
        )
    payload = {"rows": rows}
    report = build_significance_report(
        benchmark_payload=payload,
        mode="dynamic",
        n_boot=300,
        n_perm=1500,
        alpha=0.05,
        seed=123,
    )
    assert report["report_version"] == "p4_significance_v1"
    assert report["planner_count"] == 2
    assert report["pairwise_comparisons"]
    assert report["conclusions"]
    confs = {str(x.get("confidence")) for x in report["pairwise_comparisons"]}
    assert confs & {"medium", "high"}

