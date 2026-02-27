from __future__ import annotations

import itertools
import random
from math import isfinite, sqrt
from statistics import mean
from typing import Any

from app.core.p4_metrics import compute_route_safety


_METRIC_META: dict[str, dict[str, str]] = {
    "completion_rate": {"direction": "higher_better", "unit": "ratio"},
    "total_cost_km": {"direction": "lower_better", "unit": "km"},
    "replan_latency_ms": {"direction": "lower_better", "unit": "ms"},
    "risk_exposure": {"direction": "lower_better", "unit": "km_penalty"},
    "route_safety": {"direction": "higher_better", "unit": "ratio"},
}


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


def bootstrap_mean_ci(
    values: list[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float | int | None]:
    vals = [float(v) for v in values if isfinite(float(v))]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = random.Random(int(seed))
    boots: list[float] = []
    rounds = max(200, int(n_boot))
    for _ in range(rounds):
        sample = [vals[rng.randrange(0, n)] for _ in range(n)]
        boots.append(float(mean(sample)))
    boots.sort()
    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0
    return {
        "n": n,
        "mean": float(mean(vals)),
        "ci_low": _percentile(boots, lo_q),
        "ci_high": _percentile(boots, hi_q),
    }


def permutation_test_mean_diff(
    a: list[float],
    b: list[float],
    *,
    n_perm: int = 5000,
    seed: int = 42,
) -> dict[str, float | int | None]:
    vals_a = [float(x) for x in a if isfinite(float(x))]
    vals_b = [float(x) for x in b if isfinite(float(x))]
    if not vals_a or not vals_b:
        return {"n_a": len(vals_a), "n_b": len(vals_b), "mean_diff": None, "p_value": None, "effect_size": None}

    obs = float(mean(vals_a) - mean(vals_b))
    n_a = len(vals_a)
    all_vals = vals_a + vals_b
    n_all = len(all_vals)
    rng = random.Random(int(seed))
    rounds = max(1000, int(n_perm))
    extreme = 0
    for _ in range(rounds):
        perm = list(all_vals)
        rng.shuffle(perm)
        diff = mean(perm[:n_a]) - mean(perm[n_a:])
        if abs(diff) >= abs(obs):
            extreme += 1
    p_value = (extreme + 1.0) / (rounds + 1.0)

    var_a = sum((x - mean(vals_a)) ** 2 for x in vals_a) / max(1, len(vals_a) - 1)
    var_b = sum((x - mean(vals_b)) ** 2 for x in vals_b) / max(1, len(vals_b) - 1)
    pooled_std = sqrt(max(1e-12, ((len(vals_a) - 1) * var_a + (len(vals_b) - 1) * var_b) / max(1, len(vals_a) + len(vals_b) - 2)))
    effect = obs / pooled_std if pooled_std > 0 else 0.0
    return {
        "n_a": len(vals_a),
        "n_b": len(vals_b),
        "mean_diff": float(obs),
        "p_value": float(p_value),
        "effect_size": float(effect),
    }


def confidence_label(*, p_value: float | None, ci_low: float | None, ci_high: float | None) -> str:
    if p_value is None:
        return "insufficient_data"
    excludes_zero = ci_low is not None and ci_high is not None and ((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))
    if p_value < 0.01 and excludes_zero:
        return "high"
    if p_value < 0.05 and excludes_zero:
        return "medium"
    return "low"


def _collect_values_by_planner(rows: list[dict[str, Any]], *, mode: str) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("mode", "")).lower() != str(mode).lower():
            continue
        planner = str(row.get("planner", "")).strip().lower()
        if not planner:
            continue
        bucket = out.setdefault(
            planner,
            {
                "completion_rate": [],
                "total_cost_km": [],
                "replan_latency_ms": [],
                "risk_exposure": [],
                "route_safety": [],
            },
        )
        ok = str(row.get("status", "")).lower() == "ok"
        bucket["completion_rate"].append(1.0 if ok else 0.0)
        if not ok:
            continue
        bucket["total_cost_km"].append(_safe_float(row.get("route_cost_effective_km", 0.0)))
        bucket["replan_latency_ms"].append(_safe_float(row.get("avg_replan_ms", 0.0)))
        bucket["risk_exposure"].append(_safe_float(row.get("risk_exposure", 0.0)))
        if "route_safety" in row:
            bucket["route_safety"].append(_safe_float(row.get("route_safety", 0.0)))
        else:
            bucket["route_safety"].append(
                compute_route_safety(_safe_float(row.get("distance_km", 0.0)), _safe_float(row.get("caution_len_km", 0.0)))
            )
    return out


def build_significance_report(
    *,
    benchmark_payload: dict[str, Any],
    mode: str = "dynamic",
    n_boot: int = 2000,
    n_perm: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    rows = benchmark_payload.get("rows", []) if isinstance(benchmark_payload, dict) else []
    row_list = rows if isinstance(rows, list) else []
    by_planner = _collect_values_by_planner(row_list, mode=mode)

    per_planner: dict[str, dict[str, Any]] = {}
    for planner, metrics in by_planner.items():
        per_planner[planner] = {}
        for metric_name in _METRIC_META.keys():
            ci = bootstrap_mean_ci(
                metrics.get(metric_name, []),
                n_boot=int(n_boot),
                alpha=float(alpha),
                seed=int(seed),
            )
            per_planner[planner][metric_name] = ci

    comparisons: list[dict[str, Any]] = []
    planners = sorted(by_planner.keys())
    for planner_a, planner_b in itertools.combinations(planners, 2):
        for metric_name, meta in _METRIC_META.items():
            vals_a = by_planner[planner_a].get(metric_name, [])
            vals_b = by_planner[planner_b].get(metric_name, [])
            stat = permutation_test_mean_diff(vals_a, vals_b, n_perm=int(n_perm), seed=int(seed))
            mean_diff = stat.get("mean_diff")
            p_value = stat.get("p_value")
            winner = "tie"
            if mean_diff is None or p_value is None:
                winner = "insufficient_data"
            else:
                direction = meta["direction"]
                if float(p_value) < float(alpha):
                    if direction == "higher_better":
                        winner = planner_a if float(mean_diff) > 0 else planner_b
                    else:
                        winner = planner_a if float(mean_diff) < 0 else planner_b

            ci_a = per_planner.get(planner_a, {}).get(metric_name, {})
            ci_b = per_planner.get(planner_b, {}).get(metric_name, {})
            comparisons.append(
                {
                    "planner_a": planner_a,
                    "planner_b": planner_b,
                    "metric": metric_name,
                    "direction": meta["direction"],
                    "unit": meta["unit"],
                    "mean_diff_a_minus_b": mean_diff,
                    "p_value": p_value,
                    "effect_size": stat.get("effect_size"),
                    "winner": winner,
                    "confidence": confidence_label(
                        p_value=float(p_value) if p_value is not None else None,
                        ci_low=(
                            float(ci_a["ci_low"]) - float(ci_b["ci_high"])
                            if isinstance(ci_a, dict)
                            and isinstance(ci_b, dict)
                            and ci_a.get("ci_low") is not None
                            and ci_b.get("ci_high") is not None
                            else None
                        ),
                        ci_high=(
                            float(ci_a["ci_high"]) - float(ci_b["ci_low"])
                            if isinstance(ci_a, dict)
                            and isinstance(ci_b, dict)
                            and ci_a.get("ci_high") is not None
                            and ci_b.get("ci_low") is not None
                            else None
                        ),
                    ),
                    "n_a": stat.get("n_a"),
                    "n_b": stat.get("n_b"),
                }
            )

    conclusions: list[dict[str, Any]] = []
    for item in comparisons:
        metric = str(item.get("metric", ""))
        winner = str(item.get("winner", "tie"))
        p_value = item.get("p_value")
        conf = str(item.get("confidence", "low"))
        if winner == "insufficient_data":
            statement = f"Insufficient data for {metric} comparison ({item['planner_a']} vs {item['planner_b']})"
        elif winner == "tie":
            statement = f"No significant difference on {metric} ({item['planner_a']} vs {item['planner_b']})"
        else:
            statement = f"{winner} is better on {metric} ({mode})"
        conclusions.append(
            {
                "statement": statement,
                "metric": metric,
                "winner": winner,
                "p_value": p_value,
                "confidence": conf,
            }
        )

    low_sample_warnings: list[str] = []
    for planner, block in per_planner.items():
        for metric_name, ci in block.items():
            n = int(ci.get("n", 0)) if isinstance(ci, dict) else 0
            if n < 3:
                low_sample_warnings.append(f"{planner}.{metric_name}: sample_size={n} (<3)")

    return {
        "report_version": "p4_significance_v1",
        "mode": str(mode),
        "alpha": float(alpha),
        "methods": {
            "ci": {"name": "bootstrap_mean_ci", "n_boot": int(n_boot)},
            "test": {"name": "permutation_mean_diff", "n_perm": int(n_perm)},
        },
        "planner_count": len(per_planner),
        "per_planner_metrics": per_planner,
        "pairwise_comparisons": comparisons,
        "conclusions": conclusions,
        "warnings": low_sample_warnings,
    }
