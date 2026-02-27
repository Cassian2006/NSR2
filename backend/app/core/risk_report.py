from __future__ import annotations

from datetime import datetime, UTC
from typing import Any


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(candidate.get("id", "")),
        "label": str(candidate.get("label", candidate.get("id", ""))),
        "status": str(candidate.get("status", "unknown")),
        "planner": str(candidate.get("planner", "")),
        "risk_mode": str(candidate.get("risk_mode", "")),
        "caution_mode": str(candidate.get("caution_mode", "")),
        "distance_km": _to_float(candidate.get("distance_km", 0.0)),
        "risk_exposure": _to_float(candidate.get("risk_exposure", 0.0)),
        "caution_len_km": _to_float(candidate.get("caution_len_km", 0.0)),
        "corridor_score": _to_float(candidate.get("corridor_score", 0.0)),
        "pareto_rank": candidate.get("pareto_rank"),
        "pareto_frontier": bool(candidate.get("pareto_frontier", False)),
        "pareto_order": candidate.get("pareto_order"),
        "pareto_score": candidate.get("pareto_score"),
        "error": str(candidate.get("error", "")) if candidate.get("status") != "ok" else "",
    }


def build_risk_report(item: dict[str, Any]) -> dict[str, Any]:
    explain = item.get("explain", {}) if isinstance(item.get("explain"), dict) else {}
    action = item.get("action", {}) if isinstance(item.get("action"), dict) else {}
    policy = action.get("policy", {}) if isinstance(action.get("policy"), dict) else {}

    distance_km = _to_float(explain.get("distance_km", item.get("distance_km", 0.0)))
    caution_len_km = _to_float(explain.get("caution_len_km", item.get("caution_len_km", 0.0)))
    risk_exposure = _to_float(explain.get("route_cost_risk_extra_km", 0.0))
    risk_penalty_mean = _to_float(explain.get("risk_penalty_mean", 0.0))
    risk_penalty_p90 = _to_float(explain.get("risk_penalty_p90", 0.0))

    risk_constraint_mode = str(explain.get("risk_constraint_mode", "none"))
    risk_constraint_metric_name = str(explain.get("risk_constraint_metric_name", "none"))
    risk_constraint_metric = _to_float(explain.get("risk_constraint_metric", 0.0))
    if risk_constraint_metric_name == "chance_violation_ratio":
        high_risk_crossing_ratio = risk_constraint_metric
        high_risk_ratio_source = "chance_violation_ratio"
    else:
        high_risk_crossing_ratio = _to_float(explain.get("caution_cell_ratio", 0.0))
        high_risk_ratio_source = "fallback_caution_cell_ratio"

    raw_candidates = item.get("candidates") if isinstance(item.get("candidates"), list) else []
    candidates = [_normalize_candidate(c) for c in raw_candidates if isinstance(c, dict)]
    ok_candidates = [c for c in candidates if c["status"] == "ok"]

    selected = next((c for c in ok_candidates if c["id"] == "requested"), None)
    if selected is None and ok_candidates:
        selected = ok_candidates[0]

    baseline = max(ok_candidates, key=lambda c: c["risk_exposure"], default=None)
    avoidance_gain_risk = 0.0
    avoidance_tradeoff_distance_km = 0.0
    if selected is not None and baseline is not None:
        avoidance_gain_risk = float(baseline["risk_exposure"] - selected["risk_exposure"])
        avoidance_tradeoff_distance_km = float(selected["distance_km"] - baseline["distance_km"])

    report = {
        "report_version": "risk_report_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "gallery_id": str(item.get("id", "")),
        "timestamp": str(item.get("timestamp", "")),
        "summary": {
            "distance_km": round(distance_km, 3),
            "caution_len_km": round(caution_len_km, 3),
            "caution_ratio": round((caution_len_km / distance_km) if distance_km > 1e-9 else 0.0, 6),
            "corridor_alignment": round(_to_float(explain.get("corridor_alignment", 0.0)), 6),
        },
        "risk": {
            "risk_mode": str(explain.get("risk_mode", policy.get("risk_mode", "balanced"))),
            "risk_layer": str(explain.get("risk_layer", "risk_mean")),
            "risk_lambda": round(_to_float(explain.get("risk_lambda", 0.0)), 6),
            "risk_exposure": round(risk_exposure, 6),
            "risk_penalty_mean": round(risk_penalty_mean, 6),
            "risk_penalty_p90": round(risk_penalty_p90, 6),
            "high_risk_crossing_ratio": round(high_risk_crossing_ratio, 6),
            "high_risk_ratio_source": high_risk_ratio_source,
            "constraint": {
                "mode": risk_constraint_mode,
                "metric_name": risk_constraint_metric_name,
                "metric": round(risk_constraint_metric, 6),
                "budget": round(_to_float(explain.get("risk_budget", 0.0)), 6),
                "usage": round(_to_float(explain.get("risk_budget_usage", 0.0)), 6),
                "satisfied": bool(explain.get("risk_constraint_satisfied", True)),
                "confidence_level": round(_to_float(explain.get("risk_confidence_level", 0.0)), 6),
            },
            "avoidance_gain": {
                "baseline_ref": str(baseline["id"]) if baseline is not None else "",
                "selected_ref": str(selected["id"]) if selected is not None else "",
                "risk_reduction": round(avoidance_gain_risk, 6),
                "distance_tradeoff_km": round(avoidance_tradeoff_distance_km, 6),
            },
        },
        "strategy": {
            "planner": str(explain.get("planner", policy.get("planner", "astar"))),
            "caution_mode": str(explain.get("caution_mode", policy.get("caution_mode", "tie_breaker"))),
            "corridor_bias": round(_to_float(policy.get("corridor_bias", item.get("corridor_bias", 0.2))), 6),
            "blocked_sources": policy.get("blocked_sources", []),
            "smoothing": bool(policy.get("smoothing", explain.get("smoothing", True))),
        },
        "candidate_comparison": {
            "count": len(candidates),
            "ok_count": len(ok_candidates),
            "pareto_summary": item.get("pareto_summary", explain.get("pareto_summary")),
            "items": candidates,
        },
        "explain": explain,
    }
    return report

