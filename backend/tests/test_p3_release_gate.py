from __future__ import annotations

from app.core.p3_release_gate import P3GateThresholds, build_p3_release_report


def test_p3_release_gate_pass_builds_p4_handoff() -> None:
    repro = {"overall_status": "PASS"}
    benchmark = {
        "summary": {
            "static:dstar_lite": {
                "success_rate": 1.0,
                "avg_effective_cost_km": 1000.0,
                "avg_risk_exposure": 0.10,
                "avg_runtime_ms": 200.0,
            },
            "dynamic:dstar_lite": {
                "success_rate": 0.95,
                "stability_consistency": 0.9,
                "avg_risk_exposure": 0.11,
                "avg_effective_cost_km": 1080.0,
                "avg_runtime_ms": 180.0,
                "avg_replan_latency_ms": 25.0,
            },
            "dynamic:speedup": {"dstar_speedup_vs_astar_x": 1.15},
        }
    }
    runtime_profile = {
        "status": "pass",
        "runtime_monitor": {
            "step_wall_ms_mean": 500.0,
            "step_update_ms_mean": 200.0,
            "memory_peak_mb": 256.0,
            "path_metrics_cache_hit_ratio": 0.4,
        },
    }
    casebook = {"summary": {"total_cases": 3, "ok_cases": 3, "failed_cases": 0}}
    thresholds = P3GateThresholds(
        min_dynamic_success_rate=0.6,
        min_dynamic_stability=0.6,
        max_dynamic_risk_exposure=1.0,
        max_dynamic_cost_increase_pct=0.2,
        min_dstar_speedup=0.9,
        min_casebook_ok_cases=2,
    )
    report = build_p3_release_report(
        repro_summary=repro,
        benchmark_payload=benchmark,
        runtime_profile_payload=runtime_profile,
        casebook_payload=casebook,
        thresholds=thresholds,
        allow_warn_runtime=True,
        allow_warn_repro=True,
    )
    assert report["status"] == "PASS"
    assert report["checks"]["benchmark"]["status"] == "PASS"
    assert report["checks"]["runtime"]["status"] == "PASS"
    assert report["checks"]["casebook"]["status"] == "PASS"
    assert report["p4_handoff"]["protocol_version"] == "p4_eval_protocol_v1"
    assert "planner_comparison_static_vs_dynamic" in report["p4_handoff"]["report_template_sections"]


def test_p3_release_gate_fails_when_dynamic_metrics_missing() -> None:
    report = build_p3_release_report(
        repro_summary={"overall_status": "PASS"},
        benchmark_payload={"summary": {}},
        runtime_profile_payload={"status": "fail", "runtime_monitor": {}},
        casebook_payload={"summary": {"total_cases": 0, "ok_cases": 0}},
        thresholds=P3GateThresholds(),
    )
    assert report["status"] == "FAIL"
    assert report["checks"]["benchmark"]["status"] == "FAIL"
    assert report["checks"]["runtime"]["status"] == "FAIL"
    assert report["checks"]["casebook"]["status"] == "FAIL"
