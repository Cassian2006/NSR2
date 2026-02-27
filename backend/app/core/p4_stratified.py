from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from math import isfinite
from statistics import mean
from typing import Any

from app.core.p4_metrics import compute_route_safety


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if isfinite(out) else float(default)


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q <= 0.0:
        return float(sorted_vals[0])
    if q >= 1.0:
        return float(sorted_vals[-1])
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _trace_ref(row: dict[str, Any]) -> dict[str, Any]:
    mode = str(row.get("mode", ""))
    trace = {
        "mode": mode,
        "planner": str(row.get("planner", "")),
        "repeat_idx": int(_safe_float(row.get("repeat_idx", 0), 0.0)),
        "timestamp": str(row.get("timestamp", "")),
        "window_start": str(row.get("window_start", "")),
        "window_end": str(row.get("window_end", "")),
        "status": str(row.get("status", "")),
        "detail": str(row.get("detail", "")),
    }
    if mode == "dynamic":
        trace["case_ref"] = f"{trace['window_start']}->{trace['window_end']}#{trace['repeat_idx']}"
    else:
        trace["case_ref"] = f"{trace['timestamp']}#{trace['repeat_idx']}"
    return trace


def _volatility_band(row: dict[str, Any]) -> str:
    mode = str(row.get("mode", "")).lower()
    if mode != "dynamic":
        return "stable"
    replan_count = int(_safe_float(row.get("replan_count", 0), 0.0))
    avg_replan_ms = _safe_float(row.get("avg_replan_ms", 0.0), 0.0)
    return "volatile" if replan_count > 0 or avg_replan_ms > 0.0 else "stable"


def _risk_band(risk_exposure: float, threshold: float) -> str:
    return "high_risk" if float(risk_exposure) > float(threshold) else "low_risk"


def _distance_band(distance_km: float, threshold: float) -> str:
    return "long_range" if float(distance_km) > float(threshold) else "short_range"


def _metric_from_row(row: dict[str, Any], metric: str) -> float:
    if metric == "route_safety":
        if "route_safety" in row:
            return _safe_float(row.get("route_safety", 0.0), 0.0)
        return compute_route_safety(
            _safe_float(row.get("distance_km", 0.0), 0.0),
            _safe_float(row.get("caution_len_km", 0.0), 0.0),
        )
    if metric == "total_cost_km":
        return _safe_float(row.get("route_cost_effective_km", 0.0), 0.0)
    if metric == "replan_latency_ms":
        return _safe_float(row.get("avg_replan_ms", 0.0), 0.0)
    return _safe_float(row.get(metric, 0.0), 0.0)


def build_stratified_eval_report(
    *,
    benchmark_payload: dict[str, Any],
    risk_quantile: float = 0.5,
    distance_quantile: float = 0.5,
) -> dict[str, Any]:
    rows = benchmark_payload.get("rows", []) if isinstance(benchmark_payload, dict) else []
    row_list = [r for r in rows if isinstance(r, dict)]
    ok_rows = [r for r in row_list if str(r.get("status", "")).lower() == "ok"]

    risk_values = sorted(_safe_float(r.get("risk_exposure", 0.0), 0.0) for r in ok_rows)
    distance_values = sorted(_safe_float(r.get("distance_km", 0.0), 0.0) for r in ok_rows)
    risk_threshold = _percentile(risk_values, float(risk_quantile)) if risk_values else 0.0
    distance_threshold = _percentile(distance_values, float(distance_quantile)) if distance_values else 0.0

    strata_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    planner_rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    failure_index: list[dict[str, Any]] = []

    dim_values: dict[str, set[str]] = {
        "volatility_band": set(),
        "risk_band": set(),
        "distance_band": set(),
    }

    for row in row_list:
        distance_km = _safe_float(row.get("distance_km", 0.0), 0.0)
        risk_exposure = _safe_float(row.get("risk_exposure", 0.0), 0.0)
        vol = _volatility_band(row)
        risk = _risk_band(risk_exposure, risk_threshold)
        dist = _distance_band(distance_km, distance_threshold)
        key = f"{vol}|{risk}|{dist}"

        row_aug = dict(row)
        row_aug["volatility_band"] = vol
        row_aug["risk_band"] = risk
        row_aug["distance_band"] = dist
        row_aug["strata_key"] = key
        strata_rows[key].append(row_aug)
        planner = str(row.get("planner", "")).lower()
        planner_rows[key][planner].append(row_aug)

        dim_values["volatility_band"].add(vol)
        dim_values["risk_band"].add(risk)
        dim_values["distance_band"].add(dist)

        if str(row.get("status", "")).lower() != "ok":
            failure_index.append(
                {
                    "strata_key": key,
                    "volatility_band": vol,
                    "risk_band": risk,
                    "distance_band": dist,
                    "trace": _trace_ref(row),
                }
            )

    def _agg_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        ok = [x for x in items if str(x.get("status", "")).lower() == "ok"]
        succ = len(ok)
        out = {
            "sample_count": total,
            "success_count": succ,
            "fail_count": total - succ,
            "success_rate": round(float(succ / total), 6) if total > 0 else 0.0,
        }
        if not ok:
            out.update(
                {
                    "avg_distance_km": None,
                    "avg_total_cost_km": None,
                    "avg_replan_latency_ms": None,
                    "avg_risk_exposure": None,
                    "avg_route_safety": None,
                }
            )
            return out
        out.update(
            {
                "avg_distance_km": round(mean(_metric_from_row(x, "distance_km") for x in ok), 6),
                "avg_total_cost_km": round(mean(_metric_from_row(x, "total_cost_km") for x in ok), 6),
                "avg_replan_latency_ms": round(mean(_metric_from_row(x, "replan_latency_ms") for x in ok), 6),
                "avg_risk_exposure": round(mean(_metric_from_row(x, "risk_exposure") for x in ok), 6),
                "avg_route_safety": round(mean(_metric_from_row(x, "route_safety") for x in ok), 6),
            }
        )
        return out

    strata_summary: list[dict[str, Any]] = []
    planner_comparison: dict[str, list[dict[str, Any]]] = {}
    for key in sorted(strata_rows.keys()):
        items = strata_rows[key]
        vol, risk, dist = key.split("|")
        summary = _agg_metrics(items)
        strata_summary.append(
            {
                "strata_key": key,
                "volatility_band": vol,
                "risk_band": risk,
                "distance_band": dist,
                **summary,
            }
        )
        planner_block: list[dict[str, Any]] = []
        for planner, p_items in sorted(planner_rows[key].items()):
            planner_block.append({"planner": planner, **_agg_metrics(p_items)})
        planner_comparison[key] = planner_block

    expected = {
        "volatility_band": {"stable", "volatile"},
        "risk_band": {"low_risk", "high_risk"},
        "distance_band": {"short_range", "long_range"},
    }
    dimension_coverage: dict[str, dict[str, Any]] = {}
    coverage_ok = True
    for dim, expected_vals in expected.items():
        present = sorted(dim_values[dim])
        missing = sorted(expected_vals - dim_values[dim])
        if missing:
            coverage_ok = False
        dimension_coverage[dim] = {
            "present": present,
            "missing": missing,
            "ok": len(missing) == 0,
        }
    if not row_list:
        status = "FAIL"
    elif coverage_ok:
        status = "PASS"
    else:
        status = "WARN"

    charts = {
        "success_rate_by_strata_planner": [
            {
                "strata_key": key,
                "planner": item["planner"],
                "success_rate": item["success_rate"],
                "sample_count": item["sample_count"],
            }
            for key, block in planner_comparison.items()
            for item in block
        ],
        "risk_vs_cost_by_strata_planner": [
            {
                "strata_key": key,
                "planner": item["planner"],
                "avg_risk_exposure": item["avg_risk_exposure"],
                "avg_total_cost_km": item["avg_total_cost_km"],
                "sample_count": item["sample_count"],
            }
            for key, block in planner_comparison.items()
            for item in block
            if item.get("avg_risk_exposure") is not None and item.get("avg_total_cost_km") is not None
        ],
    }

    return {
        "report_version": "p4_stratified_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "risk_quantile": float(risk_quantile),
            "risk_threshold": round(float(risk_threshold), 6),
            "distance_quantile": float(distance_quantile),
            "distance_threshold": round(float(distance_threshold), 6),
        },
        "summary": {
            "status": status,
            "total_rows": len(row_list),
            "ok_rows": len(ok_rows),
            "failed_rows": len(row_list) - len(ok_rows),
            "strata_count": len(strata_summary),
            "dimension_coverage": dimension_coverage,
        },
        "strata_summary": strata_summary,
        "planner_comparison": planner_comparison,
        "failure_case_index": failure_index,
        "charts": charts,
    }

