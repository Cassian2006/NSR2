from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from app.core.config import get_settings
from app.core.dataset import get_dataset_service
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
    allow = {"astar", "dstar_lite", "any_angle", "hybrid_astar"}
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


def run_benchmark(
    *,
    start: tuple[float, float],
    goal: tuple[float, float],
    timestamps: list[str],
    planners: list[str],
    corridor_bias: float,
    blocked_sources: list[str],
) -> list[dict]:
    settings = get_settings()
    rows: list[dict] = []
    for ts in timestamps:
        for planner in planners:
            t0 = time.perf_counter()
            status = "ok"
            detail = ""
            explain = {}
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
            except PlanningError as exc:
                status = "fail"
                detail = str(exc)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rows.append(
                {
                    "mode": "static",
                    "timestamp": ts,
                    "planner": planner,
                    "status": status,
                    "runtime_ms": round(elapsed_ms, 3),
                    "distance_km": float(explain.get("distance_km", 0.0)),
                    "distance_nm": float(explain.get("distance_nm", 0.0)),
                    "caution_len_km": float(explain.get("caution_len_km", 0.0)),
                    "corridor_alignment": float(explain.get("corridor_alignment", 0.0)),
                    "raw_points": int(explain.get("raw_points", 0)),
                    "smoothed_points": int(explain.get("smoothed_points", 0)),
                    "start_adjusted": bool(explain.get("start_adjusted", False)),
                    "goal_adjusted": bool(explain.get("goal_adjusted", False)),
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
) -> list[dict]:
    settings = get_settings()
    rows: list[dict] = []
    for win_idx, timestamps in enumerate(windows):
        for planner in planners:
            t0 = time.perf_counter()
            status = "ok"
            detail = ""
            explain = {}
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
                    "status": status,
                    "timestamps": len(timestamps),
                    "advance_steps": int(advance_steps),
                    "runtime_ms": round(elapsed_ms, 3),
                    "distance_km": float(explain.get("distance_km", 0.0)),
                    "caution_len_km": float(explain.get("caution_len_km", 0.0)),
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
            summary[planner] = {
                "runs": len(items),
                "success": len(ok),
                "fail": len(fail),
                "success_rate": round(len(ok) / len(items), 4),
                "avg_runtime_ms": round(sum(x["runtime_ms"] for x in ok) / len(ok), 3),
                "avg_distance_km": round(sum(x["distance_km"] for x in ok) / len(ok), 3),
                "avg_caution_len_km": round(sum(x["caution_len_km"] for x in ok) / len(ok), 3),
                "avg_corridor_alignment": round(sum(x["corridor_alignment"] for x in ok) / len(ok), 4),
            }
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
    parser.add_argument("--dynamic-window", type=int, default=0, help=">1 to run dynamic incremental benchmark")
    parser.add_argument("--dynamic-runs", type=int, default=4, help="number of dynamic windows to benchmark")
    parser.add_argument("--advance-steps", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "benchmarks")
    args = parser.parse_args()

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
            )
        )
    summary = summarize(rows)

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
        "created_at": datetime.utcnow().isoformat() + "Z",
        "start": {"lat": start[0], "lon": start[1]},
        "goal": {"lat": goal[0], "lon": goal[1]},
        "timestamps": timestamps,
        "blocked_sources": blocked_sources,
        "summary": summary,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Benchmark complete: {csv_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
