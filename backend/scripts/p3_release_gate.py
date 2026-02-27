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

from app.core.p3_release_gate import P3GateThresholds, build_p3_release_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P3 release gate (dynamic twin acceptance + P4 handoff).")
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--sample-mode", action="store_true")
    p.add_argument("--allow-warn", action="store_true", help="Return exit 0 when gate_status=WARN.")
    p.add_argument("--repro-summary", default="")
    p.add_argument("--benchmark-json", default="")
    p.add_argument("--runtime-profile-json", default="")
    p.add_argument("--casebook-json", default="")
    p.add_argument("--run-repro-if-missing", action="store_true")
    p.add_argument("--run-benchmark-if-missing", action="store_true")
    p.add_argument("--run-runtime-profile-if-missing", action="store_true")
    p.add_argument("--run-casebook-if-missing", action="store_true")
    p.add_argument("--benchmark-out-dir", default="outputs/benchmarks")
    p.add_argument("--casebook-out-dir", default="outputs/case_library")
    p.add_argument("--out-json", default="")
    p.add_argument("--out-md", default="")
    p.add_argument("--min-dynamic-success-rate", type=float, default=None)
    p.add_argument("--min-dynamic-stability", type=float, default=None)
    p.add_argument("--max-dynamic-risk-exposure", type=float, default=None)
    p.add_argument("--max-dynamic-cost-increase-pct", type=float, default=None)
    p.add_argument("--min-dstar-speedup", type=float, default=None)
    p.add_argument("--min-casebook-ok-cases", type=int, default=None)
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


def _parse_kv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


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
        out_path = ROOT / "outputs" / "repro" / "repro_summary_p3_gate.json"
        cmd = [args.python_exe, "scripts/repro_pipeline.py", "--allow-warn", "--out", str(out_path)]
        if args.sample_mode:
            cmd.insert(3, "--sample-mode")
        run_meta = _run_cmd(cmd, cwd=ROOT)
        requested = out_path if out_path.exists() else requested
    return _load_json(requested), str(requested) if requested else "", run_meta


def _ensure_benchmark(args: argparse.Namespace) -> tuple[dict[str, Any] | None, str, dict[str, Any] | None]:
    requested = Path(args.benchmark_json).resolve() if args.benchmark_json else _find_latest_json(Path(args.benchmark_out_dir).resolve(), "planner_benchmark_*.json")
    run_meta = None
    if (requested is None or not requested.exists()) and args.run_benchmark_if_missing:
        out_dir = Path(args.benchmark_out_dir).resolve()
        cmd = [
            args.python_exe,
            "scripts/benchmark_planners.py",
            "--dynamic",
            "--planners",
            "astar,dstar_lite,any_angle",
            "--out-dir",
            str(out_dir),
            "--timestamps",
            "2" if args.sample_mode else "6",
            "--dynamic-runs",
            "1" if args.sample_mode else "3",
            "--repeats",
            "1" if args.sample_mode else "2",
        ]
        run_meta = _run_cmd(cmd, cwd=ROOT)
        requested = _find_latest_json(out_dir, "planner_benchmark_*.json")
    return _load_json(requested), str(requested) if requested else "", run_meta


def _ensure_runtime_profile(args: argparse.Namespace) -> tuple[dict[str, Any] | None, str, dict[str, Any] | None]:
    requested = Path(args.runtime_profile_json).resolve() if args.runtime_profile_json else _find_latest_json(ROOT / "outputs" / "benchmarks", "dynamic_runtime_profile_*.json")
    run_meta = None
    if (requested is None or not requested.exists()) and args.run_runtime_profile_if_missing:
        cmd = [args.python_exe, "scripts/profile_dynamic_runtime.py", "--window", "3" if args.sample_mode else "6"]
        run_meta = _run_cmd(cmd, cwd=ROOT)
        kv = _parse_kv(run_meta.get("stdout", ""))
        if kv.get("json"):
            requested = Path(kv["json"]).resolve()
        else:
            requested = _find_latest_json(ROOT / "outputs" / "benchmarks", "dynamic_runtime_profile_*.json")
    return _load_json(requested), str(requested) if requested else "", run_meta


def _ensure_casebook(args: argparse.Namespace) -> tuple[dict[str, Any] | None, str, dict[str, Any] | None]:
    requested = Path(args.casebook_json).resolve() if args.casebook_json else _find_latest_json(Path(args.casebook_out_dir).resolve(), "p3_casebook_*.json")
    run_meta = None
    if (requested is None or not requested.exists()) and args.run_casebook_if_missing:
        out_dir = Path(args.casebook_out_dir).resolve()
        cmd = [
            args.python_exe,
            "scripts/generate_p3_casebook.py",
            "--window",
            "3" if args.sample_mode else "6",
            "--case-count",
            "2" if args.sample_mode else "4",
            "--out-dir",
            str(out_dir),
        ]
        run_meta = _run_cmd(cmd, cwd=ROOT)
        kv = _parse_kv(run_meta.get("stdout", ""))
        if kv.get("json"):
            requested = Path(kv["json"]).resolve()
        else:
            requested = _find_latest_json(out_dir, "p3_casebook_*.json")
    return _load_json(requested), str(requested) if requested else "", run_meta


def _render_md(payload: dict[str, Any]) -> str:
    checks = payload.get("checks", {})
    lines = [
        "# P3 Release Gate Report",
        "",
        f"- generated_at: `{payload.get('generated_at', '')}`",
        f"- gate_status: `{payload.get('status', 'FAIL')}`",
        "",
        "## Inputs",
        f"- repro_summary: `{payload.get('inputs', {}).get('repro_summary', '')}`",
        f"- benchmark_json: `{payload.get('inputs', {}).get('benchmark_json', '')}`",
        f"- runtime_profile_json: `{payload.get('inputs', {}).get('runtime_profile_json', '')}`",
        f"- casebook_json: `{payload.get('inputs', {}).get('casebook_json', '')}`",
        "",
    ]
    for name in ("repro", "benchmark", "runtime", "casebook"):
        part = checks.get(name, {})
        lines.append(f"## {name}")
        lines.append(f"- status: `{part.get('status', 'FAIL')}`")
        for item in part.get("checks", []):
            lines.append(f"- `{item.get('name', '')}`: `{item.get('result', 'FAIL')}` ({item.get('message', '')})")
        lines.append("")
    lines.append("## Dynamic vs Static")
    for key, value in (payload.get("dynamic_vs_static", {}) or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## P4 Handoff")
    handoff = payload.get("p4_handoff", {}) or {}
    lines.append(f"- protocol_version: `{handoff.get('protocol_version', '')}`")
    lines.append(f"- core_metrics: `{handoff.get('core_metrics', [])}`")
    lines.append(f"- report_template_sections: `{handoff.get('report_template_sections', [])}`")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    thresholds = P3GateThresholds(
        min_dynamic_success_rate=(
            float(args.min_dynamic_success_rate)
            if args.min_dynamic_success_rate is not None
            else (0.55 if args.sample_mode else 0.70)
        ),
        min_dynamic_stability=(
            float(args.min_dynamic_stability)
            if args.min_dynamic_stability is not None
            else (0.50 if args.sample_mode else 0.70)
        ),
        max_dynamic_risk_exposure=(
            float(args.max_dynamic_risk_exposure)
            if args.max_dynamic_risk_exposure is not None
            else 1000.0
        ),
        max_dynamic_cost_increase_pct=(
            float(args.max_dynamic_cost_increase_pct)
            if args.max_dynamic_cost_increase_pct is not None
            else 0.40
        ),
        min_dstar_speedup=(
            float(args.min_dstar_speedup)
            if args.min_dstar_speedup is not None
            else (0.0 if args.sample_mode else 0.90)
        ),
        min_casebook_ok_cases=(
            int(args.min_casebook_ok_cases)
            if args.min_casebook_ok_cases is not None
            else (1 if args.sample_mode else 2)
        ),
    )

    repro_summary, repro_path, repro_run = _ensure_repro_summary(args)
    benchmark_payload, benchmark_path, benchmark_run = _ensure_benchmark(args)
    runtime_profile, runtime_profile_path, runtime_profile_run = _ensure_runtime_profile(args)
    casebook_payload, casebook_path, casebook_run = _ensure_casebook(args)

    report = build_p3_release_report(
        repro_summary=repro_summary,
        benchmark_payload=benchmark_payload,
        runtime_profile_payload=runtime_profile,
        casebook_payload=casebook_payload,
        thresholds=thresholds,
        allow_warn_runtime=True,
        allow_warn_repro=True,
    )
    report["generated_at"] = _utc_now_iso()
    report["inputs"] = {
        "sample_mode": bool(args.sample_mode),
        "repro_summary": repro_path,
        "benchmark_json": benchmark_path,
        "runtime_profile_json": runtime_profile_path,
        "casebook_json": casebook_path,
    }
    report["runtime"] = {
        "repro_run": repro_run,
        "benchmark_run": benchmark_run,
        "runtime_profile_run": runtime_profile_run,
        "casebook_run": casebook_run,
    }

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_json = Path(args.out_json).resolve() if args.out_json else (ROOT / "outputs" / "release" / f"p3_release_gate_{ts}.json")
    out_md = Path(args.out_md).resolve() if args.out_md else (ROOT / "outputs" / "release" / f"p3_release_gate_{ts}.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_render_md(report), encoding="utf-8")

    # Also emit a dedicated P4 handoff template for direct downstream use.
    p4_template_path = out_json.parent / f"p4_handoff_from_p3_{ts}.json"
    p4_template_path.write_text(json.dumps(report.get("p4_handoff", {}), ensure_ascii=False, indent=2), encoding="utf-8")

    gate_status = str(report.get("status", "FAIL")).upper()
    print(f"json={out_json}")
    print(f"md={out_md}")
    print(f"p4_handoff_json={p4_template_path}")
    print(f"gate_status={gate_status}")
    if gate_status == "FAIL":
        raise SystemExit(2)
    if gate_status == "WARN" and not args.allow_warn:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
