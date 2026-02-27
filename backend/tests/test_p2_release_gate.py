from __future__ import annotations

from app.core.p2_release_gate import P2GateThresholds, build_p2_release_report, evaluate_benchmark_stage


def _repro_summary_pass() -> dict:
    return {
        "overall_status": "PASS",
        "stages": [
            {"name": "contract", "status": "PASS"},
            {"name": "manifest", "status": "PASS"},
            {"name": "quality", "status": "PASS"},
            {"name": "registry", "status": "PASS"},
            {"name": "smoke_plan", "status": "PASS"},
        ],
    }


def _benchmark_summary_pass() -> dict:
    return {
        "summary": {
            "static:astar": {
                "success_rate": 0.95,
                "stability_consistency": 0.98,
                "avg_risk_exposure": 10.0,
            },
            "static:dstar_lite": {
                "success_rate": 0.90,
                "stability_consistency": 0.96,
                "avg_risk_exposure": 12.0,
            },
            "dynamic:speedup": {"dstar_speedup_vs_astar_x": 1.15},
        }
    }


def _explain_probe_pass() -> dict:
    return {
        "plan_http_status": 200,
        "risk_report_http_status": 200,
        "candidate_count": 3,
        "explain_fields": [
            "distance_km",
            "route_cost_effective_km",
            "route_cost_risk_extra_km",
            "risk_mode",
            "risk_layer",
            "smoothing_feasible",
        ],
        "candidate_fields": ["distance_km", "risk_exposure", "pareto_rank", "pareto_frontier"],
    }


def test_build_p2_release_report_pass() -> None:
    report = build_p2_release_report(
        repro_summary=_repro_summary_pass(),
        benchmark_payload=_benchmark_summary_pass(),
        explain_probe=_explain_probe_pass(),
        thresholds=P2GateThresholds(),
    )
    assert report["status"] == "PASS"
    assert report["checks"]["repro"]["status"] == "PASS"
    assert report["checks"]["benchmark"]["status"] == "PASS"
    assert report["checks"]["explainability"]["status"] == "PASS"


def test_benchmark_stage_fail_on_success_rate_regression() -> None:
    payload = {
        "summary": {
            "static:astar": {
                "success_rate": 0.40,
                "stability_consistency": 0.95,
                "avg_risk_exposure": 10.0,
            },
            "static:dstar_lite": {
                "success_rate": 0.90,
                "stability_consistency": 0.95,
                "avg_risk_exposure": 10.0,
            },
        }
    }
    stage = evaluate_benchmark_stage(payload, thresholds=P2GateThresholds(min_success_rate=0.70))
    assert stage["status"] == "FAIL"
    assert any(item.get("name") == "static:astar:success_rate" and item.get("result") == "FAIL" for item in stage["checks"])

