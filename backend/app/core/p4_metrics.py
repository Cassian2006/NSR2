from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class MetricSpec:
    id: str
    group: str
    title: str
    unit: str
    direction: str
    formula: str
    source_paths: list[str]
    required: bool = True


_CORE_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        id="completion_rate",
        group="core",
        title="完成率",
        unit="ratio",
        direction="higher_better",
        formula="success_runs / total_runs",
        source_paths=["temporal_compare.planners.<planner>.completion_rate"],
    ),
    MetricSpec(
        id="total_cost_km",
        group="core",
        title="总成本",
        unit="km",
        direction="lower_better",
        formula="avg(route_cost_effective_km)",
        source_paths=["temporal_compare.planners.<planner>.total_cost_km"],
    ),
    MetricSpec(
        id="replan_latency_ms",
        group="core",
        title="重规划时延",
        unit="ms",
        direction="lower_better",
        formula="avg(avg_replan_ms)",
        source_paths=["temporal_compare.planners.<planner>.replan_latency_ms"],
    ),
    MetricSpec(
        id="stability",
        group="core",
        title="稳定性",
        unit="ratio",
        direction="higher_better",
        formula="1 / unique(route_signature) when >0",
        source_paths=["temporal_compare.planners.<planner>.stability"],
    ),
    MetricSpec(
        id="risk_exposure",
        group="core",
        title="风险暴露",
        unit="km_penalty",
        direction="lower_better",
        formula="avg(route_cost_risk_extra_km)",
        source_paths=["temporal_compare.planners.<planner>.risk_exposure"],
    ),
    MetricSpec(
        id="route_safety",
        group="core",
        title="路线安全性",
        unit="ratio",
        direction="higher_better",
        formula="max(0, 1 - caution_len_km / max(distance_km, 1e-9))",
        source_paths=[
            "temporal_compare.planners.<planner>.route_safety",
            "summary.dynamic:<planner>.avg_route_safety",
        ],
    ),
)

_DERIVED_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        id="dynamic_vs_static_gain_pct",
        group="derived",
        title="动态相对静态收益",
        unit="ratio",
        direction="higher_better",
        formula="(static_dstar_cost - dynamic_dstar_cost) / static_dstar_cost",
        source_paths=["summary.static:dstar_lite.avg_effective_cost_km", "summary.dynamic:dstar_lite.avg_effective_cost_km"],
    ),
    MetricSpec(
        id="risk_constraint_violation_rate",
        group="derived",
        title="风险约束违约率",
        unit="ratio",
        direction="lower_better",
        formula="violations / valid_rows, violation := (risk_constraint_satisfied is False) or (risk_budget_usage > 1)",
        source_paths=["rows[*].risk_constraint_satisfied", "rows[*].risk_budget_usage"],
    ),
    MetricSpec(
        id="explain_consistency",
        group="derived",
        title="解释一致性",
        unit="ratio",
        direction="higher_better",
        formula="rows_with_required_fields / successful_rows",
        source_paths=[
            "rows[*].distance_km",
            "rows[*].caution_len_km",
            "rows[*].risk_exposure",
            "rows[*].route_cost_effective_km",
            "rows[*].corridor_alignment",
        ],
    ),
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if isfinite(out) else float(default)


def _safe_ratio(numer: float, denom: float) -> float:
    d = float(denom)
    if abs(d) <= 1e-12:
        return 0.0
    return float(numer) / d


def compute_route_safety(distance_km: float, caution_len_km: float) -> float:
    ratio = _safe_ratio(_safe_float(caution_len_km, 0.0), max(_safe_float(distance_km, 0.0), 1e-9))
    return max(0.0, 1.0 - ratio)


def metric_dictionary_payload() -> dict[str, Any]:
    return {
        "dictionary_version": "p4_metric_dictionary_v1",
        "core_metrics": [asdict(x) for x in _CORE_SPECS],
        "derived_metrics": [asdict(x) for x in _DERIVED_SPECS],
    }


def _core_specs() -> tuple[MetricSpec, ...]:
    return _CORE_SPECS


def _derived_specs() -> tuple[MetricSpec, ...]:
    return _DERIVED_SPECS


def compute_core_metrics_by_planner(*, summary: dict[str, Any], temporal_compare: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    planners_block = temporal_compare.get("planners", {}) if isinstance(temporal_compare, dict) else {}
    if not isinstance(planners_block, dict):
        planners_block = {}
    out: dict[str, dict[str, float]] = {}
    for planner, raw in planners_block.items():
        if not isinstance(raw, dict):
            continue
        metrics = dict(raw)
        route_safety_val: float | None = None
        if "route_safety" in metrics:
            route_safety_val = _safe_float(metrics.get("route_safety", 0.0))
        else:
            dyn = summary.get(f"dynamic:{planner}", {}) if isinstance(summary, dict) else {}
            if isinstance(dyn, dict) and "avg_route_safety" in dyn:
                route_safety_val = _safe_float(dyn.get("avg_route_safety", 0.0))

        def _pick(metric_name: str) -> float | None:
            if metric_name not in metrics:
                return None
            return _safe_float(metrics.get(metric_name, 0.0))

        out[str(planner)] = {
            "completion_rate": _pick("completion_rate"),
            "total_cost_km": _pick("total_cost_km"),
            "replan_latency_ms": _pick("replan_latency_ms"),
            "stability": _pick("stability"),
            "risk_exposure": _pick("risk_exposure"),
            "route_safety": route_safety_val,
        }
    return out


def compute_derived_metrics(*, summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    dynamic_dstar = summary.get("dynamic:dstar_lite", {}) if isinstance(summary, dict) else {}
    static_dstar = summary.get("static:dstar_lite", {}) if isinstance(summary, dict) else {}
    dynamic_cost = _safe_float(dynamic_dstar.get("avg_effective_cost_km", 0.0)) if isinstance(dynamic_dstar, dict) else 0.0
    static_cost = _safe_float(static_dstar.get("avg_effective_cost_km", 0.0)) if isinstance(static_dstar, dict) else 0.0
    gain_pct = _safe_ratio(static_cost - dynamic_cost, static_cost) if static_cost > 1e-9 else 0.0

    ok_rows = [r for r in rows if isinstance(r, dict) and str(r.get("status", "")).lower() == "ok"]

    valid_violation_rows = 0
    violation_count = 0
    for r in ok_rows:
        if "risk_constraint_satisfied" in r:
            valid_violation_rows += 1
            if not bool(r.get("risk_constraint_satisfied", True)):
                violation_count += 1
            continue
        if "risk_budget_usage" in r:
            valid_violation_rows += 1
            if _safe_float(r.get("risk_budget_usage", 0.0)) > 1.0:
                violation_count += 1
    violation_rate = _safe_ratio(float(violation_count), float(valid_violation_rows)) if valid_violation_rows > 0 else None

    required_fields = (
        "distance_km",
        "caution_len_km",
        "risk_exposure",
        "route_cost_effective_km",
        "corridor_alignment",
    )
    explain_ok = 0
    for r in ok_rows:
        valid = True
        for field in required_fields:
            if field not in r:
                valid = False
                break
            if not isfinite(_safe_float(r.get(field, float("nan")), float("nan"))):
                valid = False
                break
        if valid:
            explain_ok += 1
    explain_consistency = _safe_ratio(float(explain_ok), float(len(ok_rows))) if ok_rows else 0.0

    return {
        "dynamic_vs_static_gain_pct": {
            "value": round(float(gain_pct), 6),
            "dynamic_cost_km": round(float(dynamic_cost), 6),
            "static_cost_km": round(float(static_cost), 6),
        },
        "risk_constraint_violation_rate": {
            "value": round(float(violation_rate), 6) if violation_rate is not None else None,
            "violations": int(violation_count),
            "valid_rows": int(valid_violation_rows),
        },
        "explain_consistency": {
            "value": round(float(explain_consistency), 6),
            "valid_rows": int(explain_ok),
            "ok_rows": int(len(ok_rows)),
        },
    }


def validate_metric_completeness(*, summary: dict[str, Any], temporal_compare: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    core = compute_core_metrics_by_planner(summary=summary, temporal_compare=temporal_compare)
    derived = compute_derived_metrics(summary=summary, rows=rows)
    failed = False
    warned = False

    if not core:
        failed = True
        checks.append(
            {
                "name": "core_metrics_presence",
                "status": "FAIL",
                "message": "temporal_compare.planners is empty",
            }
        )
    for planner, metrics in core.items():
        for spec in _core_specs():
            val = metrics.get(spec.id)
            ok = val is not None and isfinite(_safe_float(val, float("nan")))
            if not ok:
                failed = True
            checks.append(
                {
                    "name": f"{planner}.{spec.id}",
                    "status": "PASS" if ok else "FAIL",
                    "value": val,
                }
            )

    for bounded_metric in ("completion_rate", "stability", "route_safety"):
        for planner, metrics in core.items():
            val = _safe_float(metrics.get(bounded_metric, -1.0), -1.0)
            ok = 0.0 <= val <= 1.0
            if not ok:
                failed = True
            checks.append(
                {
                    "name": f"{planner}.{bounded_metric}.range_0_1",
                    "status": "PASS" if ok else "FAIL",
                    "value": round(float(val), 6),
                }
            )

    vio = derived.get("risk_constraint_violation_rate", {})
    vio_value = vio.get("value")
    if vio_value is None:
        warned = True
        checks.append(
            {
                "name": "risk_constraint_violation_rate.available",
                "status": "WARN",
                "message": "no valid rows contain risk_constraint_satisfied or risk_budget_usage",
            }
        )
    else:
        val = _safe_float(vio_value, -1.0)
        ok = 0.0 <= val <= 1.0
        if not ok:
            failed = True
        checks.append(
            {
                "name": "risk_constraint_violation_rate.range_0_1",
                "status": "PASS" if ok else "FAIL",
                "value": round(float(val), 6),
            }
        )

    explain_consistency = _safe_float(derived.get("explain_consistency", {}).get("value", 0.0), 0.0)
    checks.append(
        {
            "name": "explain_consistency.range_0_1",
            "status": "PASS" if 0.0 <= explain_consistency <= 1.0 else "FAIL",
            "value": round(float(explain_consistency), 6),
        }
    )
    if not (0.0 <= explain_consistency <= 1.0):
        failed = True

    status = "FAIL" if failed else ("WARN" if warned else "PASS")
    return {
        "status": status,
        "checks": checks,
        "core_metric_ids": [x.id for x in _core_specs()],
        "derived_metric_ids": [x.id for x in _derived_specs()],
    }


def build_benchmark_metric_bundle(*, summary: dict[str, Any], temporal_compare: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    core = compute_core_metrics_by_planner(summary=summary, temporal_compare=temporal_compare)
    derived = compute_derived_metrics(summary=summary, rows=rows)
    validation = validate_metric_completeness(summary=summary, temporal_compare=temporal_compare, rows=rows)
    return {
        "dictionary_version": "p4_metric_dictionary_v1",
        "core_by_planner": core,
        "derived": derived,
        "validation": validation,
    }
