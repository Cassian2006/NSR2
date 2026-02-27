from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.p4_failure_casebook import build_failure_casebook


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build P4 failure/degradation casebook with root-cause tags.")
    p.add_argument("--benchmark-json", default="", help="Benchmark json path. Defaults to latest planner_benchmark_*.json.")
    p.add_argument("--runtime-degrade-pct", type=float, default=0.50)
    p.add_argument("--cost-degrade-pct", type=float, default=0.15)
    p.add_argument("--risk-degrade-pct", type=float, default=0.20)
    p.add_argument("--uncertainty-penalty-threshold", type=float, default=0.20)
    p.add_argument("--out-json", default="")
    p.add_argument("--out-md", default="")
    p.add_argument("--enforce-fail-on-case", action="store_true", help="Exit 2 when failed cases exist.")
    return p.parse_args()


def _latest_benchmark_file(out_root: Path) -> Path | None:
    candidates = [
        out_root / "benchmarks",
        ROOT / "outputs" / "benchmarks",
    ]
    files: list[Path] = []
    for c in candidates:
        if c.exists():
            files.extend(sorted(c.glob("planner_benchmark_*.json")))
    if not files:
        return None
    files = sorted(files, key=lambda p: p.stat().st_mtime)
    return files[-1]


def _to_markdown(payload: dict) -> str:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    lines: list[str] = []
    lines.append("# P4 Failure Casebook")
    lines.append("")
    lines.append(f"- status: `{summary.get('status', '')}`")
    lines.append(f"- case_count: `{summary.get('case_count', 0)}`")
    lines.append(f"- failed_count: `{summary.get('failed_count', 0)}`")
    lines.append(f"- degraded_count: `{summary.get('degraded_count', 0)}`")
    lines.append("")
    lines.append("## Root Cause Distribution")
    for tag, count in sorted((summary.get("by_tag", {}) or {}).items()):
        lines.append(f"- `{tag}`: {count}")
    if not summary.get("by_tag"):
        lines.append("- none")
    lines.append("")
    lines.append("## Cases (Top 50)")
    for item in payload.get("cases", [])[:50]:
        lines.append(
            "- `{cid}` [{typ}] planner={planner}, root={root}, scenario={scenario}".format(
                cid=item.get("case_id", ""),
                typ=item.get("case_type", ""),
                planner=item.get("planner", ""),
                root=item.get("primary_root_cause", ""),
                scenario=item.get("scenario_key", ""),
            )
        )
        detail = ((item.get("trace", {}) if isinstance(item.get("trace"), dict) else {})).get("detail", "")
        if detail:
            lines.append(f"  - detail: {detail}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    settings = get_settings()
    benchmark_file: Path | None = Path(args.benchmark_json).resolve() if args.benchmark_json else _latest_benchmark_file(settings.outputs_root)
    if benchmark_file is None or not benchmark_file.exists():
        raise SystemExit("No benchmark json found. Please run benchmark_planners.py first or pass --benchmark-json.")

    benchmark_payload = json.loads(benchmark_file.read_text(encoding="utf-8"))
    report = build_failure_casebook(
        benchmark_payload=benchmark_payload,
        runtime_degrade_pct=float(args.runtime_degrade_pct),
        cost_degrade_pct=float(args.cost_degrade_pct),
        risk_degrade_pct=float(args.risk_degrade_pct),
        uncertainty_penalty_threshold=float(args.uncertainty_penalty_threshold),
    )
    report["benchmark_file"] = str(benchmark_file)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = Path(args.out_json).resolve() if args.out_json else (settings.outputs_root / "release" / f"failure_casebook_{stamp}.json")
    out_md = Path(args.out_md).resolve() if args.out_md else (settings.outputs_root / "release" / f"failure_casebook_{stamp}.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"json={out_json}")
    print(f"md={out_md}")
    print(f"status={report.get('summary', {}).get('status', '')}")
    if args.enforce_fail_on_case and int(report.get("summary", {}).get("failed_count", 0)) > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

