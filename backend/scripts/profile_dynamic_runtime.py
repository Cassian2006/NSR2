from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.dataset import DatasetService
from app.planning.router import PlanningError, plan_grid_route_dynamic


DEFAULT_START = (70.5, 30.0)
DEFAULT_GOAL = (72.0, 150.0)


def _score_checks(
    *,
    runtime_monitor: dict[str, Any],
    max_step_wall_mean_ms: float,
    max_step_update_mean_ms: float,
    max_memory_peak_mb: float,
    min_cache_hit_ratio: float,
) -> tuple[str, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []

    def add_check(name: str, value: float, threshold: float, op: str) -> None:
        if op == "<=":
            ok = value <= threshold
        else:
            ok = value >= threshold
        checks.append(
            {
                "name": name,
                "value": float(value),
                "threshold": float(threshold),
                "operator": op,
                "status": "pass" if ok else "fail",
            }
        )

    add_check(
        "step_wall_ms_mean",
        float(runtime_monitor.get("step_wall_ms_mean", 0.0)),
        float(max_step_wall_mean_ms),
        "<=",
    )
    add_check(
        "step_update_ms_mean",
        float(runtime_monitor.get("step_update_ms_mean", 0.0)),
        float(max_step_update_mean_ms),
        "<=",
    )
    add_check(
        "memory_peak_mb",
        float(runtime_monitor.get("memory_peak_mb", 0.0)),
        float(max_memory_peak_mb),
        "<=",
    )

    cache_hits = int(runtime_monitor.get("path_metrics_cache_hits", 0))
    cache_misses = int(runtime_monitor.get("path_metrics_cache_misses", 0))
    cache_total = cache_hits + cache_misses
    cache_hit_ratio = float(runtime_monitor.get("path_metrics_cache_hit_ratio", 0.0))
    if cache_total < 5:
        checks.append(
            {
                "name": "path_metrics_cache_hit_ratio",
                "value": cache_hit_ratio,
                "threshold": float(min_cache_hit_ratio),
                "operator": ">=",
                "status": "warn",
                "note": "cache_samples_too_small",
            }
        )
    else:
        add_check(
            "path_metrics_cache_hit_ratio",
            cache_hit_ratio,
            float(min_cache_hit_ratio),
            ">=",
        )

    failed = any(c["status"] == "fail" for c in checks)
    warned = any(c["status"] == "warn" for c in checks)
    status = "fail" if failed else ("warn" if warned else "pass")
    return status, checks


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Dynamic Runtime Profile")
    lines.append("")
    lines.append(f"- status: `{report.get('status', 'unknown')}`")
    lines.append(f"- planner: `{report.get('planner', 'unknown')}`")
    lines.append(f"- timestamps: `{len(report.get('timestamps', []))}`")
    lines.append(f"- generated_at: `{report.get('generated_at', '')}`")
    lines.append("")
    lines.append("## Runtime Monitor")
    monitor = report.get("runtime_monitor", {})
    keys = [
        "state_load_mode",
        "state_load_workers",
        "state_load_wall_ms_total",
        "step_wall_ms_mean",
        "step_wall_ms_p90",
        "step_update_ms_mean",
        "step_planner_ms_mean",
        "step_metrics_ms_mean",
        "step_cpu_ms_mean",
        "cpu_wall_ratio",
        "memory_peak_mb",
        "path_metrics_cache_hits",
        "path_metrics_cache_misses",
        "path_metrics_cache_hit_ratio",
    ]
    for key in keys:
        lines.append(f"- {key}: `{monitor.get(key)}`")
    lines.append("")
    lines.append("## Checks")
    for check in report.get("checks", []):
        note = check.get("note")
        note_text = f", note={note}" if note else ""
        lines.append(
            "- [{status}] `{name}` value=`{value}` {op} `{threshold}`{note_text}".format(
                status=str(check.get("status", "unknown")).upper(),
                name=check.get("name", "unknown"),
                value=check.get("value"),
                op=check.get("operator", "?"),
                threshold=check.get("threshold"),
                note_text=note_text,
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile dynamic planning runtime metrics and export report.")
    p.add_argument("--timestamps", default="", help="Comma separated timestamps; empty means auto-select earliest window.")
    p.add_argument("--window", type=int, default=4, help="Number of timestamps when auto-selecting.")
    p.add_argument("--planner", default="dstar_lite")
    p.add_argument("--advance-steps", type=int, default=8)
    p.add_argument("--out-dir", default="")
    p.add_argument("--max-step-wall-mean-ms", type=float, default=7000.0)
    p.add_argument("--max-step-update-mean-ms", type=float, default=6500.0)
    p.add_argument("--max-memory-peak-mb", type=float, default=2048.0)
    p.add_argument("--min-cache-hit-ratio", type=float, default=0.0)
    p.add_argument("--enforce", action="store_true", help="Exit with code 2 when checks fail.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    ds = DatasetService()
    if args.timestamps.strip():
        timestamps = [ts.strip() for ts in args.timestamps.split(",") if ts.strip()]
    else:
        all_ts = ds.list_timestamps(month="all")
        if len(all_ts) < 2:
            raise SystemExit("Need at least two timestamps for dynamic profiling.")
        timestamps = all_ts[: max(2, int(args.window))]

    try:
        result = plan_grid_route_dynamic(
            settings=settings,
            timestamps=timestamps,
            start=DEFAULT_START,
            goal=DEFAULT_GOAL,
            model_version="unet_v1",
            corridor_bias=0.2,
            caution_mode="tie_breaker",
            smoothing=True,
            blocked_sources=["bathy", "unet_blocked"],
            planner=str(args.planner),
            advance_steps=max(1, int(args.advance_steps)),
            dynamic_replan_mode="on_event",
        )
    except PlanningError as exc:
        raise SystemExit(f"Dynamic profiling failed: {exc}") from exc

    explain = dict(result.explain)
    runtime_monitor = dict(explain.get("dynamic_runtime_monitor", {}))
    status, checks = _score_checks(
        runtime_monitor=runtime_monitor,
        max_step_wall_mean_ms=float(args.max_step_wall_mean_ms),
        max_step_update_mean_ms=float(args.max_step_update_mean_ms),
        max_memory_peak_mb=float(args.max_memory_peak_mb),
        min_cache_hit_ratio=float(args.min_cache_hit_ratio),
    )

    report = {
        "report_version": "dynamic_runtime_profile_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "planner": str(args.planner),
        "timestamps": list(timestamps),
        "advance_steps": int(args.advance_steps),
        "runtime_monitor": runtime_monitor,
        "checks": checks,
        "distance_km": float(explain.get("distance_km", 0.0)),
        "replan_count": int(len(explain.get("dynamic_replans", []))),
    }

    out_dir = Path(args.out_dir) if args.out_dir else (settings.outputs_root / "benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"dynamic_runtime_profile_{stamp}.json"
    md_path = out_dir / f"dynamic_runtime_profile_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    print(f"json={json_path}")
    print(f"md={md_path}")
    print(f"status={status}")
    if args.enforce and status == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
