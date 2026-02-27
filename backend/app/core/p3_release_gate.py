from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


@dataclass(frozen=True)
class P3GateThresholds:
    min_dynamic_success_rate: float = 0.70
    min_dynamic_stability: float = 0.70
    max_dynamic_risk_exposure: float = 1000.0
    max_dynamic_cost_increase_pct: float = 0.30
    min_dstar_speedup: float = 0.90
    min_casebook_ok_cases: int = 2


def evaluate_repro_stage(repro_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(repro_summary, dict):
        return {
            "status": "FAIL",
            "checks": [{"name": "repro_summary_exists", "result": "FAIL", "message": "missing repro summary"}],
        }
    overall = str(repro_summary.get("overall_status", "")).upper()
    if overall == "PASS":
        status = "PASS"
        result = "PASS"
    elif overall == "WARN":
        status = "WARN"
        result = "WARN"
    else:
        status = "FAIL"
        result = "FAIL"
    return {
        "status": status,
        "checks": [{"name": "repro_overall_status", "result": result, "message": f"overall_status={overall}"}],
        "detail": {"overall_status": overall},
    }


def evaluate_benchmark_stage(
    benchmark_payload: dict[str, Any] | None,
    *,
    thresholds: P3GateThresholds,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(benchmark_payload, dict):
        return (
            {
                "status": "FAIL",
                "checks": [{"name": "benchmark_payload_exists", "result": "FAIL", "message": "missing benchmark payload"}],
            },
            {},
        )
    summary = benchmark_payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    dynamic_dstar = summary.get("dynamic:dstar_lite", {}) if isinstance(summary.get("dynamic:dstar_lite", {}), dict) else {}
    static_dstar = summary.get("static:dstar_lite", {}) if isinstance(summary.get("static:dstar_lite", {}), dict) else {}
    speedup_block = summary.get("dynamic:speedup", {}) if isinstance(summary.get("dynamic:speedup", {}), dict) else {}
    checks: list[dict[str, Any]] = []

    success_rate = _safe_float(dynamic_dstar.get("success_rate"), 0.0)
    checks.append(
        {
            "name": "dynamic_success_rate",
            "result": "PASS" if success_rate >= thresholds.min_dynamic_success_rate else "FAIL",
            "message": f"value={success_rate:.4f}, threshold>={thresholds.min_dynamic_success_rate:.4f}",
        }
    )
    stability = _safe_float(dynamic_dstar.get("stability_consistency"), 0.0)
    checks.append(
        {
            "name": "dynamic_stability",
            "result": "PASS" if stability >= thresholds.min_dynamic_stability else "FAIL",
            "message": f"value={stability:.4f}, threshold>={thresholds.min_dynamic_stability:.4f}",
        }
    )
    dynamic_risk = _safe_float(dynamic_dstar.get("avg_risk_exposure"), 0.0)
    checks.append(
        {
            "name": "dynamic_risk_exposure",
            "result": "PASS" if dynamic_risk <= thresholds.max_dynamic_risk_exposure else "FAIL",
            "message": f"value={dynamic_risk:.4f}, threshold<={thresholds.max_dynamic_risk_exposure:.4f}",
        }
    )
    speedup = _safe_float(speedup_block.get("dstar_speedup_vs_astar_x"), 0.0)
    checks.append(
        {
            "name": "dynamic_dstar_speedup",
            "result": "PASS" if speedup >= thresholds.min_dstar_speedup else "FAIL",
            "message": f"value={speedup:.4f}, threshold>={thresholds.min_dstar_speedup:.4f}",
        }
    )

    dynamic_cost = _safe_float(dynamic_dstar.get("avg_effective_cost_km"), 0.0)
    static_cost = _safe_float(static_dstar.get("avg_effective_cost_km"), 0.0)
    if static_cost > 1e-9:
        cost_increase_pct = (dynamic_cost - static_cost) / static_cost
    else:
        cost_increase_pct = 0.0
    checks.append(
        {
            "name": "dynamic_vs_static_cost_increase_pct",
            "result": "PASS" if cost_increase_pct <= thresholds.max_dynamic_cost_increase_pct else "FAIL",
            "message": (
                f"value={cost_increase_pct:.4f}, "
                f"threshold<={thresholds.max_dynamic_cost_increase_pct:.4f}, "
                f"dynamic={dynamic_cost:.3f}, static={static_cost:.3f}"
            ),
        }
    )

    failed = any(c["result"] == "FAIL" for c in checks)
    status = "FAIL" if failed else "PASS"

    dynamic_vs_static = {
        "dynamic_dstar_avg_runtime_ms": _safe_float(dynamic_dstar.get("avg_runtime_ms"), 0.0),
        "static_dstar_avg_runtime_ms": _safe_float(static_dstar.get("avg_runtime_ms"), 0.0),
        "dynamic_dstar_avg_effective_cost_km": dynamic_cost,
        "static_dstar_avg_effective_cost_km": static_cost,
        "dynamic_dstar_avg_risk_exposure": dynamic_risk,
        "static_dstar_avg_risk_exposure": _safe_float(static_dstar.get("avg_risk_exposure"), 0.0),
        "dynamic_dstar_success_rate": success_rate,
        "static_dstar_success_rate": _safe_float(static_dstar.get("success_rate"), 0.0),
        "dynamic_dstar_replan_latency_ms": _safe_float(dynamic_dstar.get("avg_replan_latency_ms"), 0.0),
        "dynamic_dstar_speedup_vs_astar_x": speedup,
        "dynamic_vs_static_cost_increase_pct": cost_increase_pct,
    }
    return {"status": status, "checks": checks}, dynamic_vs_static


def evaluate_runtime_stage(runtime_profile_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(runtime_profile_payload, dict):
        return {
            "status": "FAIL",
            "checks": [{"name": "runtime_profile_exists", "result": "FAIL", "message": "missing runtime profile"}],
        }
    raw_status = str(runtime_profile_payload.get("status", "")).lower()
    if raw_status == "pass":
        status = "PASS"
        result = "PASS"
    elif raw_status == "warn":
        status = "WARN"
        result = "WARN"
    else:
        status = "FAIL"
        result = "FAIL"
    monitor = runtime_profile_payload.get("runtime_monitor", {})
    return {
        "status": status,
        "checks": [
            {"name": "runtime_profile_status", "result": result, "message": f"status={raw_status}"},
            {
                "name": "runtime_monitor_present",
                "result": "PASS" if isinstance(monitor, dict) and bool(monitor) else "FAIL",
                "message": f"monitor_keys={len(monitor) if isinstance(monitor, dict) else 0}",
            },
        ],
        "detail": {
            "status": raw_status,
            "step_wall_ms_mean": _safe_float(monitor.get("step_wall_ms_mean"), 0.0),
            "step_update_ms_mean": _safe_float(monitor.get("step_update_ms_mean"), 0.0),
            "memory_peak_mb": _safe_float(monitor.get("memory_peak_mb"), 0.0),
            "cache_hit_ratio": _safe_float(monitor.get("path_metrics_cache_hit_ratio"), 0.0),
        },
    }


def evaluate_casebook_stage(casebook_payload: dict[str, Any] | None, *, thresholds: P3GateThresholds) -> dict[str, Any]:
    if not isinstance(casebook_payload, dict):
        return {
            "status": "FAIL",
            "checks": [{"name": "casebook_exists", "result": "FAIL", "message": "missing casebook payload"}],
        }
    summary = casebook_payload.get("summary", {})
    ok_cases = _safe_int(summary.get("ok_cases"), 0) if isinstance(summary, dict) else 0
    total_cases = _safe_int(summary.get("total_cases"), 0) if isinstance(summary, dict) else 0
    checks = [
        {
            "name": "casebook_ok_cases",
            "result": "PASS" if ok_cases >= thresholds.min_casebook_ok_cases else "FAIL",
            "message": f"ok_cases={ok_cases}, threshold>={thresholds.min_casebook_ok_cases}",
        },
        {
            "name": "casebook_total_cases",
            "result": "PASS" if total_cases >= max(1, thresholds.min_casebook_ok_cases) else "FAIL",
            "message": f"total_cases={total_cases}",
        },
    ]
    status = "FAIL" if any(c["result"] == "FAIL" for c in checks) else "PASS"
    return {"status": status, "checks": checks, "detail": {"ok_cases": ok_cases, "total_cases": total_cases}}


def build_p4_handoff_template(
    *,
    dynamic_vs_static: dict[str, Any],
    runtime_stage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_version": "p4_eval_protocol_v1",
        "baseline_planners": ["astar", "dstar_lite", "any_angle", "hybrid_astar"],
        "dynamic_eval_required": True,
        "core_metrics": [
            "completion_rate",
            "total_cost_km",
            "replan_latency_ms",
            "stability",
            "risk_exposure",
            "route_safety",
        ],
        "p3_seed_metrics": {
            "dynamic_vs_static": dynamic_vs_static,
            "runtime_monitor": runtime_stage.get("detail", {}),
        },
        "report_template_sections": [
            "dataset_and_versioning",
            "scenario_design",
            "planner_comparison_static_vs_dynamic",
            "safety_and_risk_constraints",
            "runtime_and_resource_profile",
            "failure_cases_and_limitations",
            "reproducibility_statement",
        ],
    }


def build_p3_release_report(
    *,
    repro_summary: dict[str, Any] | None,
    benchmark_payload: dict[str, Any] | None,
    runtime_profile_payload: dict[str, Any] | None,
    casebook_payload: dict[str, Any] | None,
    thresholds: P3GateThresholds,
    allow_warn_runtime: bool = True,
    allow_warn_repro: bool = True,
) -> dict[str, Any]:
    repro_stage = evaluate_repro_stage(repro_summary)
    benchmark_stage, dynamic_vs_static = evaluate_benchmark_stage(benchmark_payload, thresholds=thresholds)
    runtime_stage = evaluate_runtime_stage(runtime_profile_payload)
    casebook_stage = evaluate_casebook_stage(casebook_payload, thresholds=thresholds)
    p4_handoff = build_p4_handoff_template(dynamic_vs_static=dynamic_vs_static, runtime_stage=runtime_stage)

    stages = {
        "repro": repro_stage,
        "benchmark": benchmark_stage,
        "runtime": runtime_stage,
        "casebook": casebook_stage,
    }

    normalized: list[str] = []
    for name, stage in stages.items():
        st = str(stage.get("status", "FAIL")).upper()
        if st == "WARN":
            if name == "runtime" and allow_warn_runtime:
                st = "PASS"
            elif name == "repro" and allow_warn_repro:
                st = "PASS"
        normalized.append(st)

    if any(s == "FAIL" for s in normalized):
        status = "FAIL"
    elif any(str(stages[name].get("status", "")).upper() == "WARN" for name in stages):
        status = "WARN"
    else:
        status = "PASS"

    stage_summary = {
        "gains": [
            "Dynamic replanning metrics are now reproducible and benchmarked against static baselines.",
            "Runtime monitor and cache hit/miss observability are integrated into explain payloads.",
            "Case library provides replay-ready dynamic examples for demo and review.",
        ],
        "limitations": [
            "Runtime profile may return WARN when cache sample size is small.",
            "Some datasets can produce WARN in repro quality gate under sample-mode.",
            "P4 statistical significance checks are not executed in P3 gate.",
        ],
        "next_focus": [
            "Freeze P4 evaluation protocol and scenario matrix from handoff template.",
            "Run full-season benchmark with fixed seeds and archive artifacts.",
            "Add confidence intervals/significance tests in P4 report flow.",
        ],
    }

    return {
        "status": status,
        "checks": stages,
        "dynamic_vs_static": dynamic_vs_static,
        "p4_handoff": p4_handoff,
        "stage_summary": stage_summary,
        "thresholds": {
            "min_dynamic_success_rate": thresholds.min_dynamic_success_rate,
            "min_dynamic_stability": thresholds.min_dynamic_stability,
            "max_dynamic_risk_exposure": thresholds.max_dynamic_risk_exposure,
            "max_dynamic_cost_increase_pct": thresholds.max_dynamic_cost_increase_pct,
            "min_dstar_speedup": thresholds.min_dstar_speedup,
            "min_casebook_ok_cases": thresholds.min_casebook_ok_cases,
        },
    }
