from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import isfinite
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if isfinite(out) else float(default)


def _scenario_key(row: dict[str, Any]) -> str:
    mode = str(row.get("mode", "")).lower()
    repeat_idx = int(_safe_float(row.get("repeat_idx", 0), 0.0))
    if mode == "dynamic":
        return f"dynamic:{row.get('window_start', '')}:{row.get('window_end', '')}:{repeat_idx}"
    return f"static:{row.get('timestamp', '')}:{repeat_idx}"


def _scenario_ref(row: dict[str, Any]) -> dict[str, Any]:
    mode = str(row.get("mode", "")).lower()
    if mode == "dynamic":
        return {
            "mode": "dynamic",
            "window_start": str(row.get("window_start", "")),
            "window_end": str(row.get("window_end", "")),
            "repeat_idx": int(_safe_float(row.get("repeat_idx", 0), 0.0)),
        }
    return {
        "mode": "static",
        "timestamp": str(row.get("timestamp", "")),
        "repeat_idx": int(_safe_float(row.get("repeat_idx", 0), 0.0)),
    }


def infer_root_cause_tags(
    row: dict[str, Any],
    *,
    degraded: bool,
    uncertainty_penalty_threshold: float,
) -> list[str]:
    tags: list[str] = []
    detail = str(row.get("detail", "")).lower()
    if any(k in detail for k in ("missing", "not found", "no data", "no timestamp", "unavailable")):
        tags.append("data_missing")

    if any(k in detail for k in ("threshold", "budget", "constraint", "chance", "cvar")):
        tags.append("threshold")
    if _safe_float(row.get("risk_budget_usage", 0.0), 0.0) > 1.0:
        tags.append("threshold")

    if any(k in detail for k in ("boundary", "outside", "out of bounds", "blocked", "land")):
        tags.append("boundary")
    if bool(row.get("start_adjusted", False)) or bool(row.get("goal_adjusted", False)):
        tags.append("boundary")

    uncertainty_penalty = _safe_float(row.get("uncertainty_penalty_mean", 0.0), 0.0)
    if uncertainty_penalty >= float(uncertainty_penalty_threshold) or "uncertainty" in detail:
        tags.append("model_uncertainty")

    if degraded:
        tags.append("planner_degradation")

    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag not in seen:
            deduped.append(tag)
            seen.add(tag)
    if not deduped:
        deduped = ["unknown"]
    return deduped


def _build_replay_hint(row: dict[str, Any], benchmark_payload: dict[str, Any]) -> dict[str, Any]:
    start = benchmark_payload.get("start", {}) if isinstance(benchmark_payload.get("start"), dict) else {}
    goal = benchmark_payload.get("goal", {}) if isinstance(benchmark_payload.get("goal"), dict) else {}
    blocked_sources = benchmark_payload.get("blocked_sources", ["bathy", "unet_blocked"])
    return {
        "endpoint": "/v1/route/plan/dynamic" if str(row.get("mode", "")).lower() == "dynamic" else "/v1/route/plan",
        "planner": str(row.get("planner", "")),
        "scenario_ref": _scenario_ref(row),
        "start": {"lat": _safe_float(start.get("lat", 70.5), 70.5), "lon": _safe_float(start.get("lon", 30.0), 30.0)},
        "goal": {"lat": _safe_float(goal.get("lat", 72.0), 72.0), "lon": _safe_float(goal.get("lon", 150.0), 150.0)},
        "policy_defaults": {
            "corridor_bias": _safe_float(benchmark_payload.get("corridor_bias", 0.2), 0.2),
            "caution_mode": "tie_breaker",
            "smoothing": True,
            "blocked_sources": blocked_sources if isinstance(blocked_sources, list) else ["bathy", "unet_blocked"],
            "planner": str(row.get("planner", "astar")),
            "risk_mode": "balanced",
            "risk_weight_scale": 1.0,
            "risk_constraint_mode": "none",
            "risk_budget": 1.0,
            "confidence_level": 0.9,
            "uncertainty_uplift": False,
            "uncertainty_uplift_scale": 1.0,
        },
    }


def build_failure_casebook(
    *,
    benchmark_payload: dict[str, Any],
    runtime_degrade_pct: float = 0.50,
    cost_degrade_pct: float = 0.15,
    risk_degrade_pct: float = 0.20,
    uncertainty_penalty_threshold: float = 0.20,
) -> dict[str, Any]:
    rows = benchmark_payload.get("rows", []) if isinstance(benchmark_payload, dict) else []
    row_list = [r for r in rows if isinstance(r, dict)]
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in row_list:
        by_scenario[_scenario_key(row)].append(row)

    cases: list[dict[str, Any]] = []
    case_counter = 1

    for row in row_list:
        status = str(row.get("status", "")).lower()
        if status == "ok":
            continue
        tags = infer_root_cause_tags(
            row,
            degraded=False,
            uncertainty_penalty_threshold=float(uncertainty_penalty_threshold),
        )
        cases.append(
            {
                "case_id": f"F{case_counter:04d}",
                "case_type": "failed",
                "status": status or "failed",
                "planner": str(row.get("planner", "")),
                "scenario_key": _scenario_key(row),
                "scenario_ref": _scenario_ref(row),
                "metrics": {
                    "runtime_ms": _safe_float(row.get("runtime_ms", 0.0), 0.0),
                    "distance_km": _safe_float(row.get("distance_km", 0.0), 0.0),
                    "total_cost_km": _safe_float(row.get("route_cost_effective_km", 0.0), 0.0),
                    "risk_exposure": _safe_float(row.get("risk_exposure", 0.0), 0.0),
                },
                "root_cause_tags": tags,
                "primary_root_cause": tags[0],
                "trace": {"detail": str(row.get("detail", "")), "raw_row": row},
                "replay_hint": _build_replay_hint(row, benchmark_payload),
            }
        )
        case_counter += 1

    for scenario_key, items in by_scenario.items():
        ok_items = [x for x in items if str(x.get("status", "")).lower() == "ok"]
        if len(ok_items) < 2:
            continue
        best_runtime = min(_safe_float(x.get("runtime_ms", 0.0), 0.0) for x in ok_items)
        best_cost = min(_safe_float(x.get("route_cost_effective_km", 0.0), 0.0) for x in ok_items)
        best_risk = min(_safe_float(x.get("risk_exposure", 0.0), 0.0) for x in ok_items)
        for row in ok_items:
            runtime = _safe_float(row.get("runtime_ms", 0.0), 0.0)
            cost = _safe_float(row.get("route_cost_effective_km", 0.0), 0.0)
            risk = _safe_float(row.get("risk_exposure", 0.0), 0.0)
            reasons: list[dict[str, Any]] = []
            if best_runtime > 1e-9:
                rt_pct = (runtime - best_runtime) / best_runtime
                if rt_pct >= float(runtime_degrade_pct):
                    reasons.append({"metric": "runtime_ms", "degrade_pct": round(float(rt_pct), 6)})
            if best_cost > 1e-9:
                cost_pct = (cost - best_cost) / best_cost
                if cost_pct >= float(cost_degrade_pct):
                    reasons.append({"metric": "total_cost_km", "degrade_pct": round(float(cost_pct), 6)})
            if best_risk > 1e-9:
                risk_pct = (risk - best_risk) / best_risk
                if risk_pct >= float(risk_degrade_pct):
                    reasons.append({"metric": "risk_exposure", "degrade_pct": round(float(risk_pct), 6)})
            if not reasons:
                continue
            tags = infer_root_cause_tags(
                row,
                degraded=True,
                uncertainty_penalty_threshold=float(uncertainty_penalty_threshold),
            )
            cases.append(
                {
                    "case_id": f"D{case_counter:04d}",
                    "case_type": "degraded",
                    "status": "ok",
                    "planner": str(row.get("planner", "")),
                    "scenario_key": scenario_key,
                    "scenario_ref": _scenario_ref(row),
                    "metrics": {
                        "runtime_ms": runtime,
                        "distance_km": _safe_float(row.get("distance_km", 0.0), 0.0),
                        "total_cost_km": cost,
                        "risk_exposure": risk,
                        "degrade_reasons": reasons,
                    },
                    "root_cause_tags": tags,
                    "primary_root_cause": tags[0],
                    "trace": {"detail": str(row.get("detail", "")), "raw_row": row},
                    "replay_hint": _build_replay_hint(row, benchmark_payload),
                }
            )
            case_counter += 1

    tag_counter: Counter[str] = Counter()
    planner_counter: Counter[str] = Counter()
    case_type_counter: Counter[str] = Counter()
    index_by_tag: dict[str, list[str]] = defaultdict(list)
    index_by_planner: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        cid = str(case.get("case_id", ""))
        planner = str(case.get("planner", ""))
        planner_counter[planner] += 1
        case_type = str(case.get("case_type", "unknown"))
        case_type_counter[case_type] += 1
        index_by_planner[planner].append(cid)
        for tag in case.get("root_cause_tags", []):
            t = str(tag)
            tag_counter[t] += 1
            index_by_tag[t].append(cid)

    summary_status = "PASS"
    if cases:
        summary_status = "WARN"
    if any(c.get("case_type") == "failed" for c in cases):
        summary_status = "FAIL"

    return {
        "report_version": "p4_failure_casebook_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_context": {
            "benchmark_version": str(benchmark_payload.get("benchmark_version", "")),
            "created_at": str(benchmark_payload.get("created_at", "")),
        },
        "thresholds": {
            "runtime_degrade_pct": float(runtime_degrade_pct),
            "cost_degrade_pct": float(cost_degrade_pct),
            "risk_degrade_pct": float(risk_degrade_pct),
            "uncertainty_penalty_threshold": float(uncertainty_penalty_threshold),
        },
        "summary": {
            "status": summary_status,
            "total_rows": len(row_list),
            "case_count": len(cases),
            "failed_count": int(case_type_counter.get("failed", 0)),
            "degraded_count": int(case_type_counter.get("degraded", 0)),
            "by_tag": dict(tag_counter),
            "by_planner": dict(planner_counter),
        },
        "index": {
            "by_tag": dict(index_by_tag),
            "by_planner": dict(index_by_planner),
        },
        "cases": cases,
    }

