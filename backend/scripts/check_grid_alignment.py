from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.grid_alignment import build_grid_alignment_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check grid alignment and write grid_spec.json.")
    p.add_argument("--sample-limit", type=int, default=80, help="How many timestamps to sample.")
    p.add_argument(
        "--grid-spec-out",
        default="",
        help="grid_spec output path. Defaults to NSR grid_spec_path.",
    )
    p.add_argument(
        "--report-out-dir",
        default="",
        help="QA report output directory. Defaults to outputs/qa.",
    )
    return p.parse_args()


def _to_markdown(report: dict) -> str:
    summary = report.get("summary", {})
    grid_spec = report.get("grid_spec", {})
    lines = [
        "# Grid Alignment Report",
        "",
        f"- status: `{summary.get('status', 'unknown')}`",
        f"- timestamp_count: `{summary.get('timestamp_count', 0)}`",
        f"- sampled_count: `{summary.get('sampled_count', 0)}`",
        f"- mismatch_count: `{summary.get('mismatch_count', 0)}`",
        f"- coverage: `{summary.get('coverage', 0.0)}`",
        "",
        "## CRS Policy",
        f"- compute: `{grid_spec.get('crs', {}).get('compute', '')}`",
        f"- display: `{grid_spec.get('crs', {}).get('display', '')}`",
        f"- web_map: `{grid_spec.get('crs', {}).get('web_map', '')}`",
        "",
        "## Grid",
        f"- shape: `{grid_spec.get('grid', {}).get('shape')}`",
        f"- bounds: `{json.dumps(grid_spec.get('grid', {}).get('bounds', {}), ensure_ascii=False)}`",
        f"- resolution: `{json.dumps(grid_spec.get('grid', {}).get('resolution', {}), ensure_ascii=False)}`",
        "",
        "## Mismatch Samples",
    ]
    for row in report.get("mismatches", [])[:20]:
        lines.append(f"- `{json.dumps(row, ensure_ascii=False)}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    report = build_grid_alignment_report(settings=settings, sample_limit=max(1, int(args.sample_limit)))

    grid_spec_out = Path(args.grid_spec_out).resolve() if args.grid_spec_out else settings.grid_spec_path.resolve()
    grid_spec_out.parent.mkdir(parents=True, exist_ok=True)
    grid_spec_out.write_text(
        json.dumps(report.get("grid_spec", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_out_dir = Path(args.report_out_dir).resolve() if args.report_out_dir else (settings.outputs_root / "qa")
    report_out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_json = report_out_dir / f"grid_alignment_{stamp}.json"
    report_md = report_out_dir / f"grid_alignment_{stamp}.md"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(_to_markdown(report), encoding="utf-8")

    print(f"grid_spec={grid_spec_out}")
    print(f"report_json={report_json}")
    print(f"report_md={report_md}")
    print(f"status={report.get('summary', {}).get('status', 'unknown')}")

    status = str(report.get("summary", {}).get("status", "unknown")).lower()
    if status == "fail":
        raise SystemExit(2)
    if status == "warn":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
