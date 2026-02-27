from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.p4_protocol import compare_repeat_rows
from app.planning.router import PlanningError, plan_grid_route, plan_grid_route_dynamic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check protocol repeatability by rerunning frozen scenarios twice.")
    p.add_argument("--protocol-json", default="")
    p.add_argument("--sample-mode", action="store_true")
    p.add_argument("--max-static", type=int, default=3)
    p.add_argument("--max-dynamic", type=int, default=2)
    p.add_argument("--planners", default="astar,dstar_lite")
    p.add_argument("--out-json", default="")
    p.add_argument("--out-md", default="")
    p.add_argument("--enforce", action="store_true", help="Exit 2 when repeatability fails.")
    return p.parse_args()


def _read_protocol(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_planners(raw: str) -> list[str]:
    vals = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return vals or ["astar", "dstar_lite"]


def _run_once(protocol: dict, planners: list[str], max_static: int, max_dynamic: int) -> list[dict]:
    settings = get_settings()
    scenarios = protocol.get("scenarios", {}) if isinstance(protocol, dict) else {}
    static_cases = list((scenarios.get("static", []) if isinstance(scenarios, dict) else []))[: max(0, int(max_static))]
    dynamic_cases = list((scenarios.get("dynamic", []) if isinstance(scenarios, dict) else []))[: max(0, int(max_dynamic))]

    rows: list[dict] = []
    for planner in planners:
        for case in static_cases:
            case_id = f"{case.get('id', 'S')}/{planner}"
            t0 = time.perf_counter()
            status = "ok"
            err = ""
            explain = {}
            try:
                res = plan_grid_route(
                    settings=settings,
                    timestamp=str(case["timestamp"]),
                    start=(float(case["start"]["lat"]), float(case["start"]["lon"])),
                    goal=(float(case["goal"]["lat"]), float(case["goal"]["lon"])),
                    model_version="unet_v1",
                    corridor_bias=0.2,
                    caution_mode="tie_breaker",
                    smoothing=True,
                    blocked_sources=["bathy", "unet_blocked"],
                    planner=str(planner),
                )
                explain = res.explain
            except (PlanningError, KeyError, TypeError, ValueError) as exc:
                status = "fail"
                err = str(exc)
            rows.append(
                {
                    "case_id": case_id,
                    "mode": "static",
                    "status": status,
                    "runtime_ms": (time.perf_counter() - t0) * 1000.0,
                    "distance_km": float(explain.get("distance_km", 0.0)),
                    "route_cost_effective_km": float(explain.get("route_cost_effective_km", 0.0)),
                    "caution_len_km": float(explain.get("caution_len_km", 0.0)),
                    "error": err,
                }
            )
        for case in dynamic_cases:
            case_id = f"{case.get('id', 'D')}/{planner}"
            t0 = time.perf_counter()
            status = "ok"
            err = ""
            explain = {}
            try:
                res = plan_grid_route_dynamic(
                    settings=settings,
                    timestamps=[str(x) for x in case["timestamps"]],
                    start=(float(case["start"]["lat"]), float(case["start"]["lon"])),
                    goal=(float(case["goal"]["lat"]), float(case["goal"]["lon"])),
                    model_version="unet_v1",
                    corridor_bias=0.2,
                    caution_mode="tie_breaker",
                    smoothing=True,
                    blocked_sources=["bathy", "unet_blocked"],
                    planner=str(planner),
                    advance_steps=int(case.get("advance_steps", 12)),
                )
                explain = res.explain
            except (PlanningError, KeyError, TypeError, ValueError) as exc:
                status = "fail"
                err = str(exc)
            rows.append(
                {
                    "case_id": case_id,
                    "mode": "dynamic",
                    "status": status,
                    "runtime_ms": (time.perf_counter() - t0) * 1000.0,
                    "distance_km": float(explain.get("distance_km", 0.0)),
                    "route_cost_effective_km": float(explain.get("route_cost_effective_km", 0.0)),
                    "caution_len_km": float(explain.get("caution_len_km", 0.0)),
                    "error": err,
                }
            )
    return rows


def _to_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# P4 Protocol Repeatability")
    lines.append("")
    lines.append(f"- status: `{report.get('status', '')}`")
    lines.append(f"- protocol_hash: `{report.get('protocol_hash', '')}`")
    lines.append(f"- case_count: `{report.get('repeatability', {}).get('case_count', 0)}`")
    lines.append(f"- abs_tol: `{report.get('repeatability', {}).get('abs_tol', 0.0)}`")
    lines.append(f"- rel_tol: `{report.get('repeatability', {}).get('rel_tol', 0.0)}`")
    lines.append("")
    lines.append("## Failures")
    fail_count = 0
    for item in report.get("repeatability", {}).get("details", []):
        bad = [m for m in item.get("metrics", []) if not bool(m.get("ok", False))]
        if not bad:
            continue
        fail_count += 1
        lines.append(f"- case `{item.get('case_id', '')}`")
        for metric in bad:
            lines.append(
                "  - `{name}`: abs_diff={abs_diff:.6f}, rel_diff={rel_diff:.6f}".format(
                    name=metric.get("name", ""),
                    abs_diff=float(metric.get("abs_diff", 0.0)),
                    rel_diff=float(metric.get("rel_diff", 0.0)),
                )
            )
    if fail_count == 0:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    settings = get_settings()
    protocol_path = (
        Path(args.protocol_json).resolve()
        if args.protocol_json
        else (settings.outputs_root / "release" / "p4_eval_protocol_v1.json")
    )
    protocol = _read_protocol(protocol_path)
    rpt = ((protocol.get("acceptance", {}) or {}).get("repeatability", {}) or {})
    metrics = [str(m) for m in rpt.get("metrics", ["distance_km", "route_cost_effective_km", "caution_len_km"])]
    abs_tol = float(rpt.get("abs_tol", 1e-6))
    rel_tol = float(rpt.get("rel_tol", 1e-6))

    planners = _parse_planners(args.planners)
    if args.sample_mode:
        max_static = max(0, min(int(args.max_static), 1))
        max_dynamic = max(0, min(int(args.max_dynamic), 1))
    else:
        max_static = max(0, int(args.max_static))
        max_dynamic = max(0, int(args.max_dynamic))

    run_a = _run_once(protocol, planners, max_static=max_static, max_dynamic=max_dynamic)
    run_b = _run_once(protocol, planners, max_static=max_static, max_dynamic=max_dynamic)
    repeatability = compare_repeat_rows(
        run_a,
        run_b,
        metrics=metrics,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    report = {
        "report_version": "p4_protocol_repeatability_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol_file": str(protocol_path),
        "protocol_hash": str(protocol.get("protocol_hash", "")),
        "status": "PASS" if repeatability.get("status") == "pass" else "FAIL",
        "planners": planners,
        "run_a_count": len(run_a),
        "run_b_count": len(run_b),
        "repeatability": repeatability,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = Path(args.out_json).resolve() if args.out_json else (settings.outputs_root / "release" / f"p4_protocol_repeatability_{stamp}.json")
    out_md = Path(args.out_md).resolve() if args.out_md else (settings.outputs_root / "release" / f"p4_protocol_repeatability_{stamp}.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"json={out_json}")
    print(f"md={out_md}")
    print(f"status={report['status']}")
    if args.enforce and report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
