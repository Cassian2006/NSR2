from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from app.core.config import get_settings
from app.core.dataset import get_dataset_service
from app.planning.router import PlanningError, plan_grid_route


def _parse_latlon(raw: str) -> tuple[float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected 'lat,lon', got: {raw}")
    return float(parts[0]), float(parts[1])


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


def run_benchmark(
    *,
    start: tuple[float, float],
    goal: tuple[float, float],
    timestamps: list[str],
    planners: list[str],
    corridor_bias: float,
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
                    blocked_sources=["bathy", "unet_blocked"],
                    planner=planner,
                )
                explain = res.explain
            except PlanningError as exc:
                status = "fail"
                detail = str(exc)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rows.append(
                {
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


def summarize(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    by_planner: dict[str, list[dict]] = {}
    for row in rows:
        by_planner.setdefault(row["planner"], []).append(row)

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
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark A* vs D* Lite on same scenarios")
    parser.add_argument("--start", type=str, default="70.5,30.0", help="start lat,lon")
    parser.add_argument("--goal", type=str, default="72.0,150.0", help="goal lat,lon")
    parser.add_argument("--timestamps", type=int, default=12, help="sample count across full timeline")
    parser.add_argument("--corridor-bias", type=float, default=0.2)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "benchmarks")
    args = parser.parse_args()

    start = _parse_latlon(args.start)
    goal = _parse_latlon(args.goal)
    timestamps = _pick_timestamps(args.timestamps)
    planners = ["astar", "dstar_lite"]
    rows = run_benchmark(
        start=start,
        goal=goal,
        timestamps=timestamps,
        planners=planners,
        corridor_bias=float(args.corridor_bias),
    )
    summary = summarize(rows)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"planner_benchmark_{ts}.csv"
    json_path = out_dir / f"planner_benchmark_{ts}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "start": {"lat": start[0], "lon": start[1]},
        "goal": {"lat": goal[0], "lon": goal[1]},
        "timestamps": timestamps,
        "summary": summary,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Benchmark complete: {csv_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
