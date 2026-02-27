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
from app.core.p4_metrics import validate_metric_completeness


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate P4 metric completeness for benchmark report payload.")
    p.add_argument("--benchmark-json", default="", help="Benchmark json path. Defaults to latest outputs/benchmarks/planner_benchmark_*.json.")
    p.add_argument("--out-json", default="")
    p.add_argument("--out-md", default="")
    p.add_argument("--enforce", action="store_true", help="Exit 2 when validation status is FAIL.")
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


def _to_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# P4 Metric Completeness")
    lines.append("")
    lines.append(f"- status: `{report.get('status', '')}`")
    lines.append(f"- benchmark_file: `{report.get('benchmark_file', '')}`")
    lines.append("")
    lines.append("## Checks")
    for item in report.get("checks", []):
        lines.append(
            "- `{name}`: {status} ({message})".format(
                name=item.get("name", ""),
                status=item.get("status", ""),
                message=item.get("message", f"value={item.get('value', '')}"),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    settings = get_settings()
    benchmark_file: Path | None = Path(args.benchmark_json).resolve() if args.benchmark_json else _latest_benchmark_file(settings.outputs_root)
    if benchmark_file is None or not benchmark_file.exists():
        raise SystemExit("No benchmark json found. Please run benchmark_planners.py first or pass --benchmark-json.")

    payload = json.loads(benchmark_file.read_text(encoding="utf-8"))
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    temporal_compare = payload.get("temporal_compare", {}) if isinstance(payload, dict) else {}
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if not isinstance(summary, dict) or not isinstance(temporal_compare, dict) or not isinstance(rows, list):
        raise SystemExit("Invalid benchmark payload structure: expected summary/temporal_compare/rows.")

    validation = validate_metric_completeness(summary=summary, temporal_compare=temporal_compare, rows=rows)
    report = {
        "report_version": "p4_metric_completeness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_file": str(benchmark_file),
        "status": validation.get("status", "FAIL"),
        "checks": validation.get("checks", []),
        "core_metric_ids": validation.get("core_metric_ids", []),
        "derived_metric_ids": validation.get("derived_metric_ids", []),
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = Path(args.out_json).resolve() if args.out_json else (settings.outputs_root / "release" / f"p4_metric_completeness_{stamp}.json")
    out_md = Path(args.out_md).resolve() if args.out_md else (settings.outputs_root / "release" / f"p4_metric_completeness_{stamp}.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"json={out_json}")
    print(f"md={out_md}")
    print(f"status={report['status']}")
    if args.enforce and str(report["status"]).upper() == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
