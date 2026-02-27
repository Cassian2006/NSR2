from __future__ import annotations

from dataclasses import dataclass
from typing import Any


STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}

DEFAULT_REQUIRED_REPRO_STAGES = ("contract", "manifest", "quality", "registry", "smoke_plan")
DEFAULT_WARNABLE_REPRO_STAGES = ("quality", "smoke_plan")
DEFAULT_REQUIRED_BENCHMARK_GROUPS = ("static:astar", "static:dstar_lite")
DEFAULT_EXPLAIN_FIELDS = (
    "distance_km",
    "route_cost_effective_km",
    "route_cost_risk_extra_km",
    "risk_mode",
    "risk_layer",
    "smoothing_feasible",
)
DEFAULT_CANDIDATE_FIELDS = ("distance_km", "risk_exposure", "pareto_rank", "pareto_frontier")


@dataclass(frozen=True)
class P2GateThresholds:
    min_success_rate: float = 0.70
    min_stability_consistency: float = 0.70
    max_avg_risk_exposure: float = 1000.0
    min_dstar_speedup: float = 0.90


def _normalize_status(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if raw in STATUS_ORDER:
        return raw
    return "FAIL"


def _merge_status(cur: str, new: str) -> str:
    cur_v = STATUS_ORDER.get(cur, 2)
    new_v = STATUS_ORDER.get(new, 2)
    return cur if cur_v >= new_v else new


def evaluate_repro_stage(
    repro_summary: dict[str, Any] | None,
    *,
    required_stages: tuple[str, ...] = DEFAULT_REQUIRED_REPRO_STAGES,
    warnable_stages: tuple[str, ...] = DEFAULT_WARNABLE_REPRO_STAGES,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    status = "PASS"
    if not isinstance(repro_summary, dict):
        return {
            "status": "FAIL",
            "checks": [{"name": "repro_summary", "status": "missing", "result": "FAIL", "message": "repro summary missing"}],
        }

    overall_status = _normalize_status(repro_summary.get("overall_status"))
    checks.append(
        {
            "name": "repro_overall",
            "status": overall_status,
            "result": "PASS" if overall_status == "PASS" else ("WARN" if overall_status == "WARN" else "FAIL"),
            "message": "overall reproducibility status",
        }
    )
    status = _merge_status(status, checks[-1]["result"])

    stage_status_map = {str(s.get("name")): _normalize_status(s.get("status")) for s in repro_summary.get("stages", []) if isinstance(s, dict)}
    warnable = set(warnable_stages)
    for stage_name in required_stages:
        stage_status = stage_status_map.get(stage_name, "FAIL")
        if stage_status == "PASS":
            result = "PASS"
            msg = "stage passed"
        elif stage_status == "WARN" and stage_name in warnable:
            result = "WARN"
            msg = "stage warning accepted for release gate"
        else:
            result = "FAIL"
            msg = "stage missing or failed"
        checks.append(
            {
                "name": f"stage:{stage_name}",
                "status": stage_status,
                "result": result,
                "message": msg,
            }
        )
        status = _merge_status(status, result)

    return {"status": status, "checks": checks}


def evaluate_benchmark_stage(
    benchmark_payload: dict[str, Any] | None,
    *,
    thresholds: P2GateThresholds,
    required_groups: tuple[str, ...] = DEFAULT_REQUIRED_BENCHMARK_GROUPS,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    status = "PASS"
    if not isinstance(benchmark_payload, dict):
        return {
            "status": "FAIL",
            "checks": [{"name": "benchmark_payload", "status": "missing", "result": "FAIL", "message": "benchmark payload missing"}],
        }

    summary = benchmark_payload.get("summary", {})
    if not isinstance(summary, dict):
        return {
            "status": "FAIL",
            "checks": [{"name": "benchmark_summary", "status": "invalid", "result": "FAIL", "message": "benchmark summary invalid"}],
        }

    for group in required_groups:
        row = summary.get(group)
        if not isinstance(row, dict):
            checks.append(
                {
                    "name": f"group:{group}",
                    "status": "missing",
                    "result": "WARN",
                    "message": "benchmark group missing",
                }
            )
            status = _merge_status(status, "WARN")
            continue

        success_rate = float(row.get("success_rate", 0.0))
        success_ok = success_rate >= float(thresholds.min_success_rate)
        checks.append(
            {
                "name": f"{group}:success_rate",
                "value": round(success_rate, 6),
                "threshold": round(float(thresholds.min_success_rate), 6),
                "result": "PASS" if success_ok else "FAIL",
                "message": "minimum success rate",
            }
        )
        status = _merge_status(status, checks[-1]["result"])

        if "stability_consistency" in row:
            stability = float(row.get("stability_consistency", 0.0))
            stable_ok = stability >= float(thresholds.min_stability_consistency)
            checks.append(
                {
                    "name": f"{group}:stability_consistency",
                    "value": round(stability, 6),
                    "threshold": round(float(thresholds.min_stability_consistency), 6),
                    "result": "PASS" if stable_ok else "FAIL",
                    "message": "minimum route stability consistency",
                }
            )
            status = _merge_status(status, checks[-1]["result"])
        else:
            checks.append(
                {
                    "name": f"{group}:stability_consistency",
                    "status": "missing",
                    "result": "WARN",
                    "message": "stability metric missing",
                }
            )
            status = _merge_status(status, "WARN")

        if "avg_risk_exposure" in row:
            risk = float(row.get("avg_risk_exposure", 0.0))
            risk_ok = risk <= float(thresholds.max_avg_risk_exposure)
            checks.append(
                {
                    "name": f"{group}:avg_risk_exposure",
                    "value": round(risk, 6),
                    "threshold": round(float(thresholds.max_avg_risk_exposure), 6),
                    "result": "PASS" if risk_ok else "FAIL",
                    "message": "maximum average risk exposure",
                }
            )
            status = _merge_status(status, checks[-1]["result"])

    speedup = summary.get("dynamic:speedup", {})
    if isinstance(speedup, dict) and "dstar_speedup_vs_astar_x" in speedup:
        speedup_val = float(speedup.get("dstar_speedup_vs_astar_x", 0.0))
        speedup_ok = speedup_val >= float(thresholds.min_dstar_speedup)
        checks.append(
            {
                "name": "dynamic:speedup:dstar_vs_astar",
                "value": round(speedup_val, 6),
                "threshold": round(float(thresholds.min_dstar_speedup), 6),
                "result": "PASS" if speedup_ok else "WARN",
                "message": "incremental planner expected speedup",
            }
        )
        status = _merge_status(status, checks[-1]["result"])
    else:
        checks.append(
            {
                "name": "dynamic:speedup:dstar_vs_astar",
                "status": "missing",
                "result": "WARN",
                "message": "dynamic speedup summary missing",
            }
        )
        status = _merge_status(status, "WARN")

    return {"status": status, "checks": checks}


def evaluate_explainability_stage(
    explain_probe: dict[str, Any] | None,
    *,
    required_explain_fields: tuple[str, ...] = DEFAULT_EXPLAIN_FIELDS,
    required_candidate_fields: tuple[str, ...] = DEFAULT_CANDIDATE_FIELDS,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    status = "PASS"
    if not isinstance(explain_probe, dict):
        return {
            "status": "FAIL",
            "checks": [{"name": "explain_probe", "status": "missing", "result": "FAIL", "message": "explain probe missing"}],
        }

    plan_status = int(explain_probe.get("plan_http_status", 0))
    plan_ok = plan_status == 200
    checks.append(
        {
            "name": "plan_http_status",
            "value": plan_status,
            "threshold": 200,
            "result": "PASS" if plan_ok else "FAIL",
            "message": "route plan endpoint should succeed",
        }
    )
    status = _merge_status(status, checks[-1]["result"])

    missing_explain = [k for k in required_explain_fields if k not in set(explain_probe.get("explain_fields", []))]
    checks.append(
        {
            "name": "explain_fields",
            "missing": missing_explain,
            "result": "PASS" if not missing_explain else "FAIL",
            "message": "required explain fields available",
        }
    )
    status = _merge_status(status, checks[-1]["result"])

    candidate_count = int(explain_probe.get("candidate_count", 0))
    checks.append(
        {
            "name": "candidate_count",
            "value": candidate_count,
            "threshold": 1,
            "result": "PASS" if candidate_count >= 1 else "FAIL",
            "message": "candidate list available",
        }
    )
    status = _merge_status(status, checks[-1]["result"])

    missing_candidate = [k for k in required_candidate_fields if k not in set(explain_probe.get("candidate_fields", []))]
    checks.append(
        {
            "name": "candidate_fields",
            "missing": missing_candidate,
            "result": "PASS" if not missing_candidate else "FAIL",
            "message": "required candidate fields available",
        }
    )
    status = _merge_status(status, checks[-1]["result"])

    risk_report_status = int(explain_probe.get("risk_report_http_status", 0))
    risk_report_ok = risk_report_status == 200
    checks.append(
        {
            "name": "risk_report_http_status",
            "value": risk_report_status,
            "threshold": 200,
            "result": "PASS" if risk_report_ok else "FAIL",
            "message": "risk report endpoint should succeed",
        }
    )
    status = _merge_status(status, checks[-1]["result"])

    return {"status": status, "checks": checks}


def build_p2_release_report(
    *,
    repro_summary: dict[str, Any] | None,
    benchmark_payload: dict[str, Any] | None,
    explain_probe: dict[str, Any] | None,
    thresholds: P2GateThresholds,
    warnable_repro_stages: tuple[str, ...] = DEFAULT_WARNABLE_REPRO_STAGES,
) -> dict[str, Any]:
    repro = evaluate_repro_stage(repro_summary, warnable_stages=warnable_repro_stages)
    benchmark = evaluate_benchmark_stage(benchmark_payload, thresholds=thresholds)
    explainability = evaluate_explainability_stage(explain_probe)

    overall = "PASS"
    for part in (repro, benchmark, explainability):
        overall = _merge_status(overall, str(part.get("status", "FAIL")))

    gains = [
        "Risk-aware planning chain has reproducibility guardrails and regression checks.",
        "Planner outputs include explainability contract and candidate Pareto context.",
        "Risk report export is available for gallery records.",
    ]
    limitations = []
    for part_name, part in [("repro", repro), ("benchmark", benchmark), ("explainability", explainability)]:
        for check in part.get("checks", []):
            result = str(check.get("result", "FAIL")).upper()
            if result in {"WARN", "FAIL"}:
                limitations.append(f"{part_name}:{check.get('name')} -> {result}")

    p3_baseline = {
        "risk_interface": ["/v1/risk/summary", "/v1/route/plan (risk policy)", "/v1/gallery/{id}/risk-report"],
        "dynamic_benchmark_anchor": "dynamic:speedup.dstar_speedup_vs_astar_x",
        "explain_contract_anchor": list(DEFAULT_EXPLAIN_FIELDS),
    }

    return {
        "status": overall,
        "thresholds": {
            "min_success_rate": thresholds.min_success_rate,
            "min_stability_consistency": thresholds.min_stability_consistency,
            "max_avg_risk_exposure": thresholds.max_avg_risk_exposure,
            "min_dstar_speedup": thresholds.min_dstar_speedup,
        },
        "checks": {
            "repro": repro,
            "benchmark": benchmark,
            "explainability": explainability,
        },
        "stage_summary": {
            "gains": gains,
            "limitations": limitations,
            "next_focus": [
                "P3 incremental replanning should keep risk contract unchanged.",
                "Improve dynamic speedup stability with larger temporal windows.",
                "Keep benchmark baseline snapshots for release regression comparisons.",
            ],
            "p3_input_baseline": p3_baseline,
        },
    }
