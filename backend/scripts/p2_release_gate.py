from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app.core.p2_release_gate import P2GateThresholds, build_p2_release_report
from app.main import app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P2 release gate (threshold + stability + explainability).")
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--sample-mode", action="store_true", help="Use lighter benchmark profile.")
    p.add_argument("--allow-warn", action="store_true", help="Return exit 0 when status=WARN.")
    p.add_argument("--repro-summary", default="", help="Path to repro summary json.")
    p.add_argument("--benchmark-json", default="", help="Path to benchmark json.")
    p.add_argument("--run-repro-if-missing", action="store_true", help="Run repro pipeline when repro summary is missing.")
    p.add_argument("--run-benchmark-if-missing", action="store_true", help="Run planner benchmark when benchmark json is missing.")
    p.add_argument("--benchmark-out-dir", default="outputs/benchmarks")
    p.add_argument("--out-json", default="", help="Output json path. Default: outputs/release/p2_release_gate_<ts>.json")
    p.add_argument("--out-md", default="", help="Output markdown path. Default: outputs/release/p2_release_gate_<ts>.md")
    p.add_argument("--min-success-rate", type=float, default=None)
    p.add_argument("--min-stability-consistency", type=float, default=None)
    p.add_argument("--max-avg-risk-exposure", type=float, default=None)
    p.add_argument("--min-dstar-speedup", type=float, default=None)
    return p.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_cmd(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }


def _find_latest_json(folder: Path, pattern: str) -> Path | None:
    if not folder.exists():
        return None
    hits = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ensure_repro_summary(args: argparse.Namespace) -> tuple[dict[str, Any] | None, str, dict[str, Any] | None]:
    requested = Path(args.repro_summary).resolve() if args.repro_summary else _find_latest_json(ROOT / "outputs" / "repro", "repro_summary*.json")
    run_meta = None
    if (requested is None or not requested.exists()) and args.run_repro_if_missing:
        auto_path = ROOT / "outputs" / "repro" / "repro_summary_p2_gate.json"
        cmd = [args.python_exe, "scripts/repro_pipeline.py", "--allow-warn", "--out", str(auto_path)]
        if args.sample_mode:
            cmd.insert(3, "--sample-mode")
        run_meta = _run_cmd(cmd, cwd=ROOT)
        requested = auto_path if auto_path.exists() else requested
    return _load_json(requested), str(requested) if requested else "", run_meta


def _ensure_benchmark(args: argparse.Namespace) -> tuple[dict[str, Any] | None, str, dict[str, Any] | None]:
    requested = Path(args.benchmark_json).resolve() if args.benchmark_json else _find_latest_json(Path(args.benchmark_out_dir).resolve(), "planner_benchmark_*.json")
    run_meta = None
    if (requested is None or not requested.exists()) and args.run_benchmark_if_missing:
        out_dir = Path(args.benchmark_out_dir).resolve()
        cmd = [
            args.python_exe,
            "scripts/benchmark_planners.py",
            "--planners",
            "astar,dstar_lite",
            "--out-dir",
            str(out_dir),
            "--timestamps",
            "2" if args.sample_mode else "6",
            "--repeats",
            "1" if args.sample_mode else "2",
        ]
        run_meta = _run_cmd(cmd, cwd=ROOT)
        requested = _find_latest_json(out_dir, "planner_benchmark_*.json")
    return _load_json(requested), str(requested) if requested else "", run_meta


def _probe_explainability() -> dict[str, Any]:
    with TestClient(app) as client:
        ts_resp = client.get("/v1/timestamps")
        if ts_resp.status_code != 200:
            return {
                "plan_http_status": int(ts_resp.status_code),
                "risk_report_http_status": 0,
                "candidate_count": 0,
                "explain_fields": [],
                "candidate_fields": [],
            }
        timestamps = [str(x) for x in ts_resp.json().get("timestamps", [])]
        for ts in timestamps[:16]:
            payload = {
                "timestamp": ts,
                "start": {"lat": 70.5, "lon": 30.0},
                "goal": {"lat": 72.0, "lon": 150.0},
                "policy": {
                    "objective": "shortest_distance_under_safety",
                    "blocked_sources": ["bathy"],
                    "caution_mode": "tie_breaker",
                    "corridor_bias": 0.2,
                    "smoothing": True,
                    "planner": "astar",
                    "risk_mode": "balanced",
                    "risk_weight_scale": 1.0,
                    "return_candidates": True,
                    "candidate_limit": 3,
                },
            }
            plan_resp = client.post("/v1/route/plan", json=payload)
            if plan_resp.status_code != 200:
                continue
            body = plan_resp.json()
            explain = body.get("explain", {}) if isinstance(body, dict) else {}
            candidates = body.get("candidates", []) if isinstance(body, dict) else []
            first_candidate = candidates[0] if isinstance(candidates, list) and candidates else {}
            gallery_id = str(body.get("gallery_id", "")) if isinstance(body, dict) else ""

            risk_report_status = 0
            if gallery_id:
                rr_resp = client.get(f"/v1/gallery/{gallery_id}/risk-report")
                risk_report_status = int(rr_resp.status_code)
            return {
                "plan_http_status": int(plan_resp.status_code),
                "risk_report_http_status": int(risk_report_status),
                "candidate_count": len(candidates) if isinstance(candidates, list) else 0,
                "explain_fields": sorted(explain.keys()) if isinstance(explain, dict) else [],
                "candidate_fields": sorted(first_candidate.keys()) if isinstance(first_candidate, dict) else [],
                "timestamp": ts,
                "gallery_id": gallery_id,
            }
        return {
            "plan_http_status": 422,
            "risk_report_http_status": 0,
            "candidate_count": 0,
            "explain_fields": [],
            "candidate_fields": [],
        }


def _render_md(payload: dict[str, Any]) -> str:
    checks = payload.get("checks", {})
    lines = [
        "# P2 Release Gate Report",
        "",
        f"- generated_at: `{payload.get('generated_at', '')}`",
        f"- gate_status: `{payload.get('status', 'FAIL')}`",
        "",
        "## Inputs",
        f"- repro_summary: `{payload.get('inputs', {}).get('repro_summary', '')}`",
        f"- benchmark_json: `{payload.get('inputs', {}).get('benchmark_json', '')}`",
        "",
    ]
    for name in ("repro", "benchmark", "explainability"):
        part = checks.get(name, {})
        lines.append(f"## {name}")
        lines.append(f"- status: `{part.get('status', 'FAIL')}`")
        for item in part.get("checks", []):
            lines.append(
                f"- `{item.get('name', '')}`: `{item.get('result', 'FAIL')}`"
                f" ({item.get('message', '')})"
            )
        lines.append("")
    stage_summary = payload.get("stage_summary", {})
    lines.append("## Stage Summary")
    for key in ("gains", "limitations", "next_focus"):
        vals = stage_summary.get(key, [])
        lines.append(f"- {key}:")
        if vals:
            for v in vals:
                lines.append(f"  - {v}")
        else:
            lines.append("  - (none)")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    min_success_rate = float(args.min_success_rate) if args.min_success_rate is not None else (0.60 if args.sample_mode else 0.70)
    min_stability = (
        float(args.min_stability_consistency)
        if args.min_stability_consistency is not None
        else (0.50 if args.sample_mode else 0.70)
    )
    max_risk = float(args.max_avg_risk_exposure) if args.max_avg_risk_exposure is not None else 1000.0
    min_speedup = float(args.min_dstar_speedup) if args.min_dstar_speedup is not None else (0.0 if args.sample_mode else 0.90)

    thresholds = P2GateThresholds(
        min_success_rate=min_success_rate,
        min_stability_consistency=min_stability,
        max_avg_risk_exposure=max_risk,
        min_dstar_speedup=min_speedup,
    )

    repro_summary, repro_path, repro_run = _ensure_repro_summary(args)
    benchmark_payload, benchmark_path, bench_run = _ensure_benchmark(args)
    explain_probe = _probe_explainability()
    report = build_p2_release_report(
        repro_summary=repro_summary,
        benchmark_payload=benchmark_payload,
        explain_probe=explain_probe,
        thresholds=thresholds,
        warnable_repro_stages=("contract", "quality", "smoke_plan") if args.sample_mode else ("quality", "smoke_plan"),
    )
    report["generated_at"] = _utc_now_iso()
    report["inputs"] = {
        "sample_mode": bool(args.sample_mode),
        "repro_summary": repro_path,
        "benchmark_json": benchmark_path,
    }
    report["runtime"] = {
        "repro_run": repro_run,
        "benchmark_run": bench_run,
    }

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_json = Path(args.out_json).resolve() if args.out_json else (ROOT / "outputs" / "release" / f"p2_release_gate_{ts}.json")
    out_md = Path(args.out_md).resolve() if args.out_md else (ROOT / "outputs" / "release" / f"p2_release_gate_{ts}.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_render_md(report), encoding="utf-8")

    gate_status = str(report.get("status", "FAIL")).upper()
    print(f"json={out_json}")
    print(f"md={out_md}")
    print(f"gate_status={gate_status}")
    if gate_status == "FAIL":
        raise SystemExit(2)
    if gate_status == "WARN" and not args.allow_warn:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
