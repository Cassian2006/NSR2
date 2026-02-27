from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.dataset import get_dataset_service
from app.core.p4_metrics import build_benchmark_metric_bundle, compute_route_safety
from app.planning.router import PlanningError, plan_grid_route, plan_grid_route_dynamic


def _parse_latlon(raw: str) -> tuple[float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected 'lat,lon', got: {raw}")
    return float(parts[0]), float(parts[1])


def _parse_blocked_sources(raw: str) -> list[str]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items or ["bathy", "unet_blocked"]


def _parse_planners(raw: str) -> list[str]:
    allow = {"astar", "dstar_lite", "dstar_lite_recompute", "any_angle", "hybrid_astar"}
    planners: list[str] = []
    for item in raw.split(","):
        val = item.strip().lower()
        if not val:
            continue
        if val not in allow:
            raise ValueError(f"Unsupported planner '{val}', expected one of {sorted(allow)}")
        if val not in planners:
            planners.append(val)
    if not planners:
        raise ValueError("At least one planner is required.")
    return planners


def _pick_timestamps(limit: int) -> list[str]:
    service = get_dataset_service()
    all_ts = service.list_timestamps(month="all")
    if not all_ts:
        raise RuntimeError("No timestamps found in dataset")
    if limit <= 0 or limit >= len(all_ts):
        return all_ts
    step = max(1, len(all_ts) // limit)
    picks = all_ts[::step][:limit]
    if all_ts[-1] not in picks:
        picks.append(all_ts[-1])
    return picks


def _pick_dynamic_windows(all_ts: list[str], window: int, runs: int) -> list[list[str]]:
    if window < 2:
        return []
    if len(all_ts) < window:
        raise RuntimeError(f"Not enough timestamps for dynamic window={window}")
    runs = max(1, int(runs))
    if runs == 1:
        return [all_ts[:window]]
    max_start = len(all_ts) - window
    if max_start <= 0:
        return [all_ts[:window]]
    stride = max(1, max_start // max(1, runs - 1))
    starts = [min(max_start, i * stride) for i in range(runs)]
    unique_starts = []
    seen = set()
    for s in starts:
        if s not in seen:
            seen.add(s)
            unique_starts.append(s)
    return [all_ts[s : s + window] for s in unique_starts]


def _route_signature(route_geojson: dict | None) -> str:
    coords = (route_geojson or {}).get("geometry", {}).get("coordinates", [])
    if not isinstance(coords, list) or not coords:
        return ""
    rounded: list[tuple[float, float]] = []
    for item in coords:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        lon = round(float(item[0]), 5)
        lat = round(float(item[1]), 5)
        rounded.append((lon, lat))
    if not rounded:
        return ""
    raw = json.dumps(rounded, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _safe_pct_change(new: float, old: float) -> float:
    if abs(old) <= 1e-12:
        return 0.0 if abs(new) <= 1e-12 else 1.0
    return (new - old) / old


def run_benchmark(
    *,
    start: tuple[float, float],
    goal: tuple[float, float],
    timestamps: list[str],
    planners: list[str],
    corridor_bias: float,
    blocked_sources: list[str],
    repeats: int,
) -> list[dict]:
    settings = get_settings()
    rows: list[dict] = []
    for ts in timestamps:
        for planner in planners:
            for repeat_idx in range(max(1, int(repeats))):
                t0 = time.perf_counter()
                status = "ok"
                detail = ""
                explain = {}
                route_signature = ""
                route_safety = 0.0
                try:
                    res = plan_grid_route(
                        settings=settings,
                        timestamp=ts,
                        start=start,
                        goal=goal,
                        model_version="unet_v1",
                        corridor_bias=corridor_bias,
                        caution_mode="tie_breaker",
                        smoothing=True,
                        blocked_sources=blocked_sources,
                        planner=planner,
                    )
                    explain = res.explain
                    route_signature = _route_signature(res.route_geojson)
                    route_safety = compute_route_safety(
                        float(explain.get("distance_km", 0.0)),
                        float(explain.get("caution_len_km", 0.0)),
                    )
                except PlanningError as exc:
                    status = "fail"
                    detail = str(exc)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                rows.append(
                    {
                        "mode": "static",
                        "timestamp": ts,
                        "planner": planner,
                        "repeat_idx": int(repeat_idx),
                        "status": status,
                        "runtime_ms": round(elapsed_ms, 3),
                        "distance_km": float(explain.get("distance_km", 0.0)),
                        "distance_nm": float(explain.get("distance_nm", 0.0)),
                        "caution_len_km": float(explain.get("caution_len_km", 0.0)),
                        "risk_exposure": float(explain.get("route_cost_risk_extra_km", 0.0)),
                        "route_cost_effective_km": float(explain.get("route_cost_effective_km", 0.0)),
                        "route_safety": float(route_safety),
                        "risk_budget_usage": float(explain.get("risk_budget_usage", 0.0)),
                        "risk_constraint_satisfied": bool(explain.get("risk_constraint_satisfied", True)),
                        "corridor_alignment": float(explain.get("corridor_alignment", 0.0)),
                        "raw_points": int(explain.get("raw_points", 0)),
                        "smoothed_points": int(explain.get("smoothed_points", 0)),
                        "start_adjusted": bool(explain.get("start_adjusted", False)),
                        "goal_adjusted": bool(explain.get("goal_adjusted", False)),
                        "route_signature": route_signature,
                        "detail": detail,
                    }
                )
    return rows


def run_dynamic_benchmark(
    *,
    start: tuple[float, float],
    goal: tuple[float, float],
    windows: list[list[str]],
    planners: list[str],
    corridor_bias: float,
    advance_steps: int,
    blocked_sources: list[str],
    repeats: int,
) -> list[dict]:
    settings = get_settings()
    rows: list[dict] = []
    for win_idx, timestamps in enumerate(windows):
        for planner in planners:
            for repeat_idx in range(max(1, int(repeats))):
                t0 = time.perf_counter()
                status = "ok"
                detail = ""
                explain = {}
                route_signature = ""
                route_safety = 0.0
                try:
                    res = plan_grid_route_dynamic(
                        settings=settings,
                        timestamps=timestamps,
                        start=start,
                        goal=goal,
                        model_version="unet_v1",
                        corridor_bias=corridor_bias,
                        caution_mode="tie_breaker",
                        smoothing=True,
                        blocked_sources=blocked_sources,
                        planner=planner,
                        advance_steps=advance_steps,
                    )
                    explain = res.explain
                    route_signature = _route_signature(res.route_geojson)
                    route_safety = compute_route_safety(
                        float(explain.get("distance_km", 0.0)),
                        float(explain.get("caution_len_km", 0.0)),
                    )
                except PlanningError as exc:
                    status = "fail"
                    detail = str(exc)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                replans = explain.get("dynamic_replans", []) if isinstance(explain, dict) else []
                rows.append(
                    {
                        "mode": "dynamic",
                        "window_idx": int(win_idx),
                        "window_start": timestamps[0],
                        "window_end": timestamps[-1],
                        "planner": planner,
                        "repeat_idx": int(repeat_idx),
                        "status": status,
                        "timestamps": len(timestamps),
                        "advance_steps": int(advance_steps),
                        "runtime_ms": round(elapsed_ms, 3),
                        "distance_km": float(explain.get("distance_km", 0.0)),
                        "caution_len_km": float(explain.get("caution_len_km", 0.0)),
                        "risk_exposure": float(explain.get("route_cost_risk_extra_km", 0.0)),
                        "route_cost_effective_km": float(explain.get("route_cost_effective_km", 0.0)),
                        "route_safety": float(route_safety),
                        "risk_budget_usage": float(explain.get("risk_budget_usage", 0.0)),
                        "risk_constraint_satisfied": bool(explain.get("risk_constraint_satisfied", True)),
                        "corridor_alignment": float(explain.get("corridor_alignment", 0.0)),
                        "replan_count": len(replans) if isinstance(replans, list) else 0,
                        "avg_replan_ms": round(
                            float(sum(float(x.get("runtime_ms", 0.0)) for x in replans) / max(1, len(replans))),
                            3,
                        )
                        if isinstance(replans, list) and replans
                        else 0.0,
                        "incremental_steps": int(explain.get("dynamic_incremental_steps", 0)),
                        "rebuild_steps": int(explain.get("dynamic_rebuild_steps", 0)),
                        "route_signature": route_signature,
                        "detail": detail,
                    }
                )
    return rows


def summarize(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    by_planner: dict[str, list[dict]] = {}
    for row in rows:
        mode = str(row.get("mode", "static"))
        group = f"{mode}:{row['planner']}"
        by_planner.setdefault(group, []).append(row)

    for planner, items in by_planner.items():
        ok = [x for x in items if x["status"] == "ok"]
        fail = [x for x in items if x["status"] != "ok"]
        if ok:
            runtimes = [float(x.get("runtime_ms", 0.0)) for x in ok]
            distances = [float(x.get("distance_km", 0.0)) for x in ok]
            cautions = [float(x.get("caution_len_km", 0.0)) for x in ok]
            aligns = [float(x.get("corridor_alignment", 0.0)) for x in ok]
            risks = [float(x.get("risk_exposure", 0.0)) for x in ok]
            eff_costs = [float(x.get("route_cost_effective_km", 0.0)) for x in ok]
            safeties = [float(x.get("route_safety", compute_route_safety(x.get("distance_km", 0.0), x.get("caution_len_km", 0.0)))) for x in ok]
            replan_latencies = [float(x.get("avg_replan_ms", 0.0)) for x in ok if "avg_replan_ms" in x]
            signatures = {str(x.get("route_signature", "")) for x in ok if str(x.get("route_signature", ""))}
            signature_unique = len(signatures)
            stability_consistency = 1.0 if signature_unique <= 1 else 1.0 / float(signature_unique)
            row_summary = {
                "runs": len(items),
                "success": len(ok),
                "fail": len(fail),
                "success_rate": round(len(ok) / len(items), 4),
                "avg_runtime_ms": round(mean(runtimes), 3),
                "avg_distance_km": round(mean(distances), 3),
                "avg_caution_len_km": round(mean(cautions), 3),
                "avg_corridor_alignment": round(mean(aligns), 4),
                "avg_risk_exposure": round(mean(risks), 4),
                "avg_effective_cost_km": round(mean(eff_costs), 3),
                "avg_route_safety": round(mean(safeties), 6),
                "distance_std_km": round(pstdev(distances), 6) if len(distances) > 1 else 0.0,
                "risk_std": round(pstdev(risks), 6) if len(risks) > 1 else 0.0,
                "runtime_std_ms": round(pstdev(runtimes), 6) if len(runtimes) > 1 else 0.0,
                "route_signature_unique": signature_unique,
                "stability_consistency": round(stability_consistency, 4),
            }
            if replan_latencies:
                row_summary["avg_replan_latency_ms"] = round(mean(replan_latencies), 3)
                row_summary["replan_latency_std_ms"] = round(pstdev(replan_latencies), 6) if len(replan_latencies) > 1 else 0.0
            summary[planner] = row_summary
        else:
            summary[planner] = {
                "runs": len(items),
                "success": 0,
                "fail": len(fail),
                "success_rate": 0.0,
            }

    dynamic_astar = summary.get("dynamic:astar", {})
    dynamic_dstar = summary.get("dynamic:dstar_lite", {})
    if dynamic_astar.get("avg_runtime_ms") and dynamic_dstar.get("avg_runtime_ms"):
        astar_rt = float(dynamic_astar["avg_runtime_ms"])
        dstar_rt = float(dynamic_dstar["avg_runtime_ms"])
        speedup = (astar_rt / dstar_rt) if dstar_rt > 0 else 0.0
        summary["dynamic:speedup"] = {
            "astar_avg_runtime_ms": round(astar_rt, 3),
            "dstar_avg_runtime_ms": round(dstar_rt, 3),
            "dstar_speedup_vs_astar_x": round(speedup, 3),
            "dstar_runtime_reduction_pct": round((1.0 - dstar_rt / astar_rt) * 100.0, 2) if astar_rt > 0 else 0.0,
        }
    return summary


def build_temporal_compare(*, summary: dict, planners: list[str]) -> dict:
    out: dict[str, dict] = {}
    for planner in planners:
        key = f"dynamic:{planner}"
        group = summary.get(key)
        if not isinstance(group, dict):
            continue
        out[planner] = {
            "completion_rate": round(float(group.get("success_rate", 0.0)), 6),
            "total_cost_km": round(float(group.get("avg_effective_cost_km", 0.0)), 6),
            "replan_latency_ms": round(float(group.get("avg_replan_latency_ms", 0.0)), 6),
            "stability": round(float(group.get("stability_consistency", 0.0)), 6),
            "distance_km": round(float(group.get("avg_distance_km", 0.0)), 6),
            "risk_exposure": round(float(group.get("avg_risk_exposure", 0.0)), 6),
            "route_safety": round(float(group.get("avg_route_safety", 0.0)), 6),
            "success": int(group.get("success", 0)),
            "runs": int(group.get("runs", 0)),
        }

    ranking: list[dict] = []
    if out:
        def _score(item: dict) -> float:
            # higher is better: completion/stability; lower is better: cost/latency/risk
            return (
                2.0 * float(item["completion_rate"])
                + 1.2 * float(item["stability"])
                - 0.0005 * float(item["replan_latency_ms"])
                - 0.0010 * float(item["total_cost_km"])
                - 0.0020 * float(item["risk_exposure"])
                + 0.8 * float(item.get("route_safety", 0.0))
            )

        ranked = sorted(out.items(), key=lambda kv: _score(kv[1]), reverse=True)
        for idx, (planner, metrics) in enumerate(ranked, start=1):
            ranking.append(
                {
                    "planner": planner,
                    "rank": idx,
                    "score": round(float(_score(metrics)), 6),
                    "completion_rate": metrics["completion_rate"],
                    "total_cost_km": metrics["total_cost_km"],
                    "replan_latency_ms": metrics["replan_latency_ms"],
                    "stability": metrics["stability"],
                    "route_safety": metrics["route_safety"],
                }
            )

    return {
        "benchmark_version": "temporal_benchmark_v1",
        "axes": ["completion_rate", "total_cost_km", "replan_latency_ms", "stability", "route_safety"],
        "planners": out,
        "ranking": ranking,
    }


def evaluate_regression(
    *,
    current_summary: dict,
    baseline_summary: dict,
    runtime_max_increase_pct: float = 0.35,
    distance_max_increase_pct: float = 0.15,
    risk_max_increase_pct: float = 0.20,
    success_rate_max_drop: float = 0.05,
    stability_consistency_max_drop: float = 0.10,
) -> dict:
    compared: list[dict] = []
    status = "pass"
    for key, cur in current_summary.items():
        if key not in baseline_summary:
            continue
        base = baseline_summary[key]
        if not isinstance(cur, dict) or not isinstance(base, dict):
            continue
        if float(cur.get("success_rate", 0.0)) <= 0 and float(base.get("success_rate", 0.0)) <= 0:
            continue
        detail = {"planner": key, "checks": [], "status": "pass"}

        def _check_increase(metric: str, threshold: float) -> None:
            cur_v = float(cur.get(metric, 0.0))
            base_v = float(base.get(metric, 0.0))
            pct = _safe_pct_change(cur_v, base_v)
            hit = pct > threshold
            detail["checks"].append(
                {
                    "metric": metric,
                    "baseline": round(base_v, 6),
                    "current": round(cur_v, 6),
                    "delta_pct": round(pct, 6),
                    "threshold": round(float(threshold), 6),
                    "violation": hit,
                }
            )
            if hit:
                detail["status"] = "fail"

        def _check_drop(metric: str, threshold: float) -> None:
            cur_v = float(cur.get(metric, 0.0))
            base_v = float(base.get(metric, 0.0))
            drop = float(base_v - cur_v)
            hit = drop > threshold
            detail["checks"].append(
                {
                    "metric": metric,
                    "baseline": round(base_v, 6),
                    "current": round(cur_v, 6),
                    "drop": round(drop, 6),
                    "threshold": round(float(threshold), 6),
                    "violation": hit,
                }
            )
            if hit:
                detail["status"] = "fail"

        _check_increase("avg_runtime_ms", runtime_max_increase_pct)
        _check_increase("avg_distance_km", distance_max_increase_pct)
        _check_increase("avg_risk_exposure", risk_max_increase_pct)
        _check_drop("success_rate", success_rate_max_drop)
        _check_drop("stability_consistency", stability_consistency_max_drop)

        if detail["status"] == "fail":
            status = "fail"
        compared.append(detail)

    return {
        "status": status if compared else "no_baseline",
        "thresholds": {
            "runtime_max_increase_pct": runtime_max_increase_pct,
            "distance_max_increase_pct": distance_max_increase_pct,
            "risk_max_increase_pct": risk_max_increase_pct,
            "success_rate_max_drop": success_rate_max_drop,
            "stability_consistency_max_drop": stability_consistency_max_drop,
        },
        "compared": compared,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multiple planners on same scenarios")
    parser.add_argument("--start", type=str, default="70.5,30.0", help="start lat,lon")
    parser.add_argument("--goal", type=str, default="72.0,150.0", help="goal lat,lon")
    parser.add_argument("--timestamps", type=int, default=12, help="sample count across full timeline")
    parser.add_argument("--corridor-bias", type=float, default=0.2)
    parser.add_argument(
        "--planners",
        type=str,
        default="astar,dstar_lite,any_angle,hybrid_astar",
        help="comma-separated planners, e.g. astar,dstar_lite,any_angle,hybrid_astar",
    )
    parser.add_argument(
        "--blocked-sources",
        type=str,
        default="bathy,unet_blocked",
        help="comma-separated blocked sources, e.g. bathy or bathy,unet_blocked",
    )
    parser.add_argument("--repeats", type=int, default=2, help="repeat each scenario for stability metrics")
    parser.add_argument("--dynamic-window", type=int, default=0, help=">1 to run dynamic incremental benchmark")
    parser.add_argument("--dynamic", action="store_true", help="enable dynamic benchmark with default window/runs when omitted")
    parser.add_argument("--dynamic-runs", type=int, default=4, help="number of dynamic windows to benchmark")
    parser.add_argument("--advance-steps", type=int, default=12)
    parser.add_argument("--baseline-json", type=Path, default=None, help="optional baseline benchmark json for regression check")
    parser.add_argument("--enforce-regression", action="store_true", help="exit with code 2 when regression status is fail")
    parser.add_argument("--runtime-max-increase-pct", type=float, default=0.35)
    parser.add_argument("--distance-max-increase-pct", type=float, default=0.15)
    parser.add_argument("--risk-max-increase-pct", type=float, default=0.20)
    parser.add_argument("--success-rate-max-drop", type=float, default=0.05)
    parser.add_argument("--stability-consistency-max-drop", type=float, default=0.10)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "benchmarks")
    args = parser.parse_args()

    if args.dynamic and int(args.dynamic_window) <= 1:
        args.dynamic_window = 4
    if args.dynamic and int(args.dynamic_runs) <= 0:
        args.dynamic_runs = 4

    start = _parse_latlon(args.start)
    goal = _parse_latlon(args.goal)
    blocked_sources = _parse_blocked_sources(args.blocked_sources)
    planners = _parse_planners(args.planners)
    timestamps = _pick_timestamps(args.timestamps)
    rows = run_benchmark(
        start=start,
        goal=goal,
        timestamps=timestamps,
        planners=planners,
        corridor_bias=float(args.corridor_bias),
        blocked_sources=blocked_sources,
        repeats=int(args.repeats),
    )
    if int(args.dynamic_window) > 1:
        all_ts = get_dataset_service().list_timestamps(month="all")
        windows = _pick_dynamic_windows(all_ts, int(args.dynamic_window), int(args.dynamic_runs))
        rows.extend(
            run_dynamic_benchmark(
                start=start,
                goal=goal,
                windows=windows,
                planners=planners,
                corridor_bias=float(args.corridor_bias),
                advance_steps=int(args.advance_steps),
                blocked_sources=blocked_sources,
                repeats=int(args.repeats),
            )
        )
    summary = summarize(rows)
    temporal_compare = build_temporal_compare(summary=summary, planners=planners)

    regression = None
    if args.baseline_json:
        baseline_payload = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        baseline_summary = baseline_payload.get("summary", {}) if isinstance(baseline_payload, dict) else {}
        regression = evaluate_regression(
            current_summary=summary,
            baseline_summary=baseline_summary,
            runtime_max_increase_pct=float(args.runtime_max_increase_pct),
            distance_max_increase_pct=float(args.distance_max_increase_pct),
            risk_max_increase_pct=float(args.risk_max_increase_pct),
            success_rate_max_drop=float(args.success_rate_max_drop),
            stability_consistency_max_drop=float(args.stability_consistency_max_drop),
        )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"planner_benchmark_{ts}.csv"
    json_path = out_dir / f"planner_benchmark_{ts}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "benchmark_version": "temporal_benchmark_v1",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "start": {"lat": start[0], "lon": start[1]},
        "goal": {"lat": goal[0], "lon": goal[1]},
        "timestamps": timestamps,
        "blocked_sources": blocked_sources,
        "repeats": int(args.repeats),
        "axes": ["completion_rate", "total_cost_km", "replan_latency_ms", "stability"],
        "summary": summary,
        "temporal_compare": temporal_compare,
        "regression": regression,
        "rows": rows,
    }
    payload["p4_metrics"] = build_benchmark_metric_bundle(
        summary=payload["summary"],
        temporal_compare=payload["temporal_compare"],
        rows=payload["rows"],
    )
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Benchmark complete: {csv_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Temporal compare:")
    print(json.dumps(temporal_compare, ensure_ascii=False, indent=2))
    if regression is not None:
        print("Regression check:")
        print(json.dumps(regression, ensure_ascii=False, indent=2))
        if args.enforce_regression and regression.get("status") == "fail":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
