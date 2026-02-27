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
from app.core.dataset import get_dataset_service
from app.planning.router import PlanningError, plan_grid_route_dynamic


DEFAULT_START = (70.5, 30.0)
DEFAULT_GOAL = (72.0, 150.0)


def pick_case_windows(all_timestamps: list[str], *, window: int, case_count: int) -> list[list[str]]:
    if len(all_timestamps) < max(2, window):
        return []
    window = max(2, int(window))
    case_count = max(1, int(case_count))
    max_start = len(all_timestamps) - window
    if max_start <= 0:
        return [all_timestamps[:window]]
    if case_count == 1:
        starts = [0]
    else:
        stride = max(1, max_start // max(1, case_count - 1))
        starts = [min(max_start, idx * stride) for idx in range(case_count)]
    seen: set[int] = set()
    out: list[list[str]] = []
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        out.append(all_timestamps[start : start + window])
    return out


def _render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# P3 Dynamic Casebook")
    lines.append("")
    lines.append(f"- generated_at: `{payload.get('generated_at', '')}`")
    lines.append(f"- status: `{payload.get('status', 'unknown')}`")
    summary = payload.get("summary", {})
    lines.append(f"- total_cases: `{summary.get('total_cases', 0)}`")
    lines.append(f"- ok_cases: `{summary.get('ok_cases', 0)}`")
    lines.append(f"- failed_cases: `{summary.get('failed_cases', 0)}`")
    lines.append("")
    lines.append("## Cases")
    for case in payload.get("cases", []):
        cid = case.get("case_id", "")
        st = case.get("status", "unknown")
        lines.append(f"### {cid} ({st})")
        lines.append(f"- planner: `{case.get('planner', '')}`")
        lines.append(f"- window: `{case.get('window_start', '')}` -> `{case.get('window_end', '')}`")
        if st == "ok":
            lines.append(f"- distance_km: `{case.get('distance_km', 0.0)}`")
            lines.append(f"- caution_len_km: `{case.get('caution_len_km', 0.0)}`")
            lines.append(f"- replan_count: `{case.get('replan_count', 0)}`")
            lines.append(f"- runtime_ms_total: `{case.get('runtime_ms_total', 0.0)}`")
            lines.append(f"- runtime_step_wall_ms_mean: `{case.get('runtime_step_wall_ms_mean', 0.0)}`")
        else:
            lines.append(f"- error: `{case.get('error', '')}`")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate P3 dynamic planner casebook (demo + replay artifacts).")
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--case-count", type=int, default=3)
    p.add_argument("--planner", default="dstar_lite")
    p.add_argument("--advance-steps", type=int, default=8)
    p.add_argument("--out-dir", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    timestamps = get_dataset_service().list_timestamps(month="all")
    windows = pick_case_windows(timestamps, window=int(args.window), case_count=int(args.case_count))
    if not windows:
        raise SystemExit("No enough timestamps to generate casebook windows.")

    cases: list[dict[str, Any]] = []
    for idx, win in enumerate(windows, start=1):
        case_id = f"p3_case_{idx:02d}"
        try:
            result = plan_grid_route_dynamic(
                settings=settings,
                timestamps=win,
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
            explain = dict(result.explain)
            runtime = explain.get("dynamic_runtime_monitor", {})
            cases.append(
                {
                    "case_id": case_id,
                    "status": "ok",
                    "planner": str(args.planner),
                    "window_start": win[0],
                    "window_end": win[-1],
                    "timestamp_count": len(win),
                    "distance_km": float(explain.get("distance_km", 0.0)),
                    "caution_len_km": float(explain.get("caution_len_km", 0.0)),
                    "risk_extra_km": float(explain.get("route_cost_risk_extra_km", 0.0)),
                    "route_cost_effective_km": float(explain.get("route_cost_effective_km", 0.0)),
                    "replan_count": int(len(explain.get("dynamic_replans", []))),
                    "runtime_ms_total": float(explain.get("replan_runtime_ms_total", 0.0)),
                    "runtime_step_wall_ms_mean": float(runtime.get("step_wall_ms_mean", 0.0)),
                    "runtime_step_update_ms_mean": float(runtime.get("step_update_ms_mean", 0.0)),
                    "runtime_memory_peak_mb": float(runtime.get("memory_peak_mb", 0.0)),
                    "cache_hit_ratio": float(runtime.get("path_metrics_cache_hit_ratio", 0.0)),
                    "dynamic_state_available": bool(explain.get("dynamic_replay_ready", False)),
                }
            )
        except PlanningError as exc:
            cases.append(
                {
                    "case_id": case_id,
                    "status": "failed",
                    "planner": str(args.planner),
                    "window_start": win[0],
                    "window_end": win[-1],
                    "timestamp_count": len(win),
                    "error": str(exc),
                }
            )

    ok_cases = [c for c in cases if c.get("status") == "ok"]
    payload = {
        "report_version": "p3_casebook_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if ok_cases else "fail",
        "summary": {
            "total_cases": len(cases),
            "ok_cases": len(ok_cases),
            "failed_cases": len(cases) - len(ok_cases),
        },
        "cases": cases,
    }

    out_dir = Path(args.out_dir) if args.out_dir else (settings.outputs_root / "case_library")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"p3_casebook_{stamp}.json"
    md_path = out_dir / f"p3_casebook_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"json={json_path}")
    print(f"md={md_path}")
    print(f"status={payload['status']}")


if __name__ == "__main__":
    main()
