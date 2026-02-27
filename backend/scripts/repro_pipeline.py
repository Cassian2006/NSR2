from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app.main import app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-click reproducibility pipeline.")
    p.add_argument("--sample-mode", action="store_true", help="Use lighter checks for CI smoke.")
    p.add_argument(
        "--out",
        default="",
        help="Output summary path. Defaults to outputs/repro/repro_summary.json",
    )
    p.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used for subprocess stages.",
    )
    p.add_argument(
        "--allow-warn",
        action="store_true",
        help="Return success (exit 0) when overall status is WARN.",
    )
    p.add_argument(
        "--with-p2-gate",
        action="store_true",
        help="Run P2 release gate stage after smoke checks.",
    )
    p.add_argument(
        "--p2-gate-allow-warn",
        action="store_true",
        help="Allow WARN status in P2 gate stage.",
    )
    p.add_argument(
        "--with-p3-gate",
        action="store_true",
        help="Run P3 release gate stage after smoke checks.",
    )
    p.add_argument(
        "--p3-gate-allow-warn",
        action="store_true",
        help="Allow WARN status in P3 gate stage.",
    )
    return p.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_kv_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _run_cmd(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    ended = time.perf_counter()
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "duration_sec": round(ended - started, 3),
        "stdout": stdout,
        "stderr": stderr,
        "kv": _parse_kv_lines(stdout),
    }


def _status_from_contract(code: int) -> str:
    if code == 0:
        return "PASS"
    if code == 1:
        return "WARN"
    return "FAIL"


def _status_from_gate(gate_status: str | None) -> str:
    if not gate_status:
        return "FAIL"
    v = gate_status.strip().upper()
    if v in {"PASS", "WARN", "FAIL"}:
        return v
    return "FAIL"


def _stage_contract(*, args: argparse.Namespace) -> dict[str, Any]:
    sample_limit = "12" if args.sample_mode else "20"
    result = _run_cmd(
        [
            args.python_exe,
            "scripts/check_dataset_contract.py",
            "--sample-limit",
            sample_limit,
        ],
        cwd=ROOT,
    )
    status = _status_from_contract(result["returncode"])
    return {
        "name": "contract",
        "status": status,
        "detail": {
            "json_report": result["kv"].get("json", ""),
            "md_report": result["kv"].get("md", ""),
            "raw_status": result["kv"].get("status", ""),
        },
        "exec": result,
    }


def _stage_manifest(*, args: argparse.Namespace) -> dict[str, Any]:
    result = _run_cmd(
        [
            args.python_exe,
            "scripts/build_data_manifest.py",
        ],
        cwd=ROOT,
    )
    status = "PASS" if result["returncode"] == 0 else "FAIL"
    return {
        "name": "manifest",
        "status": status,
        "detail": {
            "manifest": result["kv"].get("manifest", ""),
            "state": result["kv"].get("state", ""),
            "summary": result["kv"].get("summary", ""),
            "stats": result["kv"].get("stats", ""),
        },
        "exec": result,
    }


def _stage_quality(*, args: argparse.Namespace) -> dict[str, Any]:
    sample_limit = "80" if args.sample_mode else "120"
    result = _run_cmd(
        [
            args.python_exe,
            "scripts/report_data_quality.py",
            "--sample-limit",
            sample_limit,
        ],
        cwd=ROOT,
    )
    gate_status = result["kv"].get("gate_status", "").upper()
    status = _status_from_gate(gate_status)
    if args.sample_mode and status == "FAIL":
        # Demo/CI sample datasets may have known timeline gaps; keep smoke non-blocking.
        status = "WARN"
    return {
        "name": "quality",
        "status": status,
        "detail": {
            "json_report": result["kv"].get("json", ""),
            "md_report": result["kv"].get("md", ""),
            "gate_status": gate_status,
            "raw_status": result["kv"].get("status", ""),
        },
        "exec": result,
    }


def _stage_registry(*, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    with TestClient(app) as client:
        resp = client.get("/v1/datasets/registry", params={"page": 1, "page_size": 5})
    ended = time.perf_counter()
    status = "PASS" if resp.status_code == 200 else "FAIL"
    payload: dict[str, Any] = {}
    if resp.status_code == 200:
        payload = resp.json()
    return {
        "name": "registry",
        "status": status,
        "detail": {
            "http_status": int(resp.status_code),
            "total_samples": int(payload.get("summary", {}).get("total_samples", 0)) if payload else 0,
            "complete_rate": float(payload.get("summary", {}).get("complete_rate", 0.0)) if payload else 0.0,
            "data_version": str(payload.get("summary", {}).get("data_version", "")) if payload else "",
        },
        "exec": {
            "duration_sec": round(ended - started, 3),
        },
    }


def _stage_smoke_plan(*, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    with TestClient(app) as client:
        ts_resp = client.get("/v1/timestamps")
        if ts_resp.status_code != 200:
            ended = time.perf_counter()
            return {
                "name": "smoke_plan",
                "status": "FAIL",
                "detail": {"reason": "timestamps_endpoint_failed", "http_status": int(ts_resp.status_code)},
                "exec": {"duration_sec": round(ended - started, 3)},
            }
        timestamps = ts_resp.json().get("timestamps", [])
        if not timestamps:
            ended = time.perf_counter()
            return {
                "name": "smoke_plan",
                "status": "FAIL",
                "detail": {"reason": "no_timestamps_available"},
                "exec": {"duration_sec": round(ended - started, 3)},
            }
        ts = str(timestamps[0])
        payload = {
            "timestamp": ts,
            "start": {"lat": 70.5, "lon": 30.0},
            "goal": {"lat": 72.0, "lon": 150.0},
            "policy": {
                "objective": "shortest_distance_under_safety",
                "blocked_sources": ["bathy"],
                "caution_mode": "budget",
                "corridor_bias": 0.2,
                "smoothing": True,
                "planner": "astar",
            },
        }
        plan_resp = client.post("/v1/route/plan", json=payload)
    ended = time.perf_counter()
    if plan_resp.status_code != 200:
        if (
            args.sample_mode
            and plan_resp.status_code == 422
            and "No feasible route found" in plan_resp.text
        ):
            return {
                "name": "smoke_plan",
                "status": "WARN",
                "detail": {
                    "http_status": int(plan_resp.status_code),
                    "reason": "no_feasible_route_in_sample_mode",
                    "body": plan_resp.text[:500],
                },
                "exec": {"duration_sec": round(ended - started, 3)},
            }
        return {
            "name": "smoke_plan",
            "status": "FAIL",
            "detail": {"http_status": int(plan_resp.status_code), "body": plan_resp.text[:500]},
            "exec": {"duration_sec": round(ended - started, 3)},
        }
    body = plan_resp.json()
    return {
        "name": "smoke_plan",
        "status": "PASS",
        "detail": {
            "timestamp": ts,
            "gallery_id": str(body.get("gallery_id", "")),
            "distance_km": float(body.get("explain", {}).get("distance_km", 0.0)),
            "run_snapshot_id": str(body.get("run_snapshot_id", "")),
        },
        "exec": {"duration_sec": round(ended - started, 3)},
    }


def _stage_p2_gate(*, args: argparse.Namespace) -> dict[str, Any]:
    out_json = ROOT / "outputs" / "release" / "p2_release_gate_from_repro.json"
    out_md = ROOT / "outputs" / "release" / "p2_release_gate_from_repro.md"
    cmd = [
        args.python_exe,
        "scripts/p2_release_gate.py",
        "--run-repro-if-missing",
        "--benchmark-out-dir",
        str(ROOT / "outputs" / "benchmarks"),
        "--run-benchmark-if-missing",
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    if args.sample_mode:
        cmd.append("--sample-mode")
    if args.p2_gate_allow_warn:
        cmd.append("--allow-warn")

    result = _run_cmd(cmd, cwd=ROOT)
    code = int(result.get("returncode", 1))
    if code == 0:
        status = "PASS"
    elif code == 1:
        status = "WARN"
    else:
        status = "FAIL"
    return {
        "name": "p2_gate",
        "status": status,
        "detail": {
            "json_report": result["kv"].get("json", ""),
            "md_report": result["kv"].get("md", ""),
            "gate_status": result["kv"].get("gate_status", ""),
        },
        "exec": result,
    }


def _stage_p3_gate(*, args: argparse.Namespace) -> dict[str, Any]:
    out_json = ROOT / "outputs" / "release" / "p3_release_gate_from_repro.json"
    out_md = ROOT / "outputs" / "release" / "p3_release_gate_from_repro.md"
    cmd = [
        args.python_exe,
        "scripts/p3_release_gate.py",
        "--run-repro-if-missing",
        "--run-benchmark-if-missing",
        "--run-runtime-profile-if-missing",
        "--run-casebook-if-missing",
        "--benchmark-out-dir",
        str(ROOT / "outputs" / "benchmarks"),
        "--casebook-out-dir",
        str(ROOT / "outputs" / "case_library"),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    if args.sample_mode:
        cmd.append("--sample-mode")
    if args.p3_gate_allow_warn:
        cmd.append("--allow-warn")
    result = _run_cmd(cmd, cwd=ROOT)
    code = int(result.get("returncode", 1))
    if code == 0:
        status = "PASS"
    elif code == 1:
        status = "WARN"
    else:
        status = "FAIL"
    return {
        "name": "p3_gate",
        "status": status,
        "detail": {
            "json_report": result["kv"].get("json", ""),
            "md_report": result["kv"].get("md", ""),
            "gate_status": result["kv"].get("gate_status", ""),
            "p4_handoff_json": result["kv"].get("p4_handoff_json", ""),
        },
        "exec": result,
    }


def _overall_status(stages: list[dict[str, Any]]) -> str:
    statuses = [str(s.get("status", "FAIL")).upper() for s in stages]
    if any(s == "FAIL" for s in statuses):
        return "FAIL"
    if any(s == "WARN" for s in statuses):
        return "WARN"
    return "PASS"


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    stages = [
        _stage_contract(args=args),
        _stage_manifest(args=args),
        _stage_quality(args=args),
        _stage_registry(args=args),
        _stage_smoke_plan(args=args),
    ]
    if args.with_p2_gate:
        stages.append(_stage_p2_gate(args=args))
    if args.with_p3_gate:
        stages.append(_stage_p3_gate(args=args))

    overall = _overall_status(stages)
    ended = time.perf_counter()

    default_out = ROOT / "outputs" / "repro" / "repro_summary.json"
    out_path = Path(args.out).resolve() if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": _utc_now_iso(),
        "sample_mode": bool(args.sample_mode),
        "overall_status": overall,
        "duration_sec": round(ended - started, 3),
        "stages": stages,
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"summary={out_path}")
    print(f"overall_status={overall}")
    for st in stages:
        print(f"stage.{st['name']}={st['status']}")

    if overall == "FAIL":
        raise SystemExit(2)
    if overall == "WARN" and not args.allow_warn:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
