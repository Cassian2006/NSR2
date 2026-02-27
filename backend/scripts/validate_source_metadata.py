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
from app.core.source_metadata import build_source_metadata_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate per-timestamp source metadata coverage.")
    p.add_argument(
        "--sample-limit",
        type=int,
        default=200,
        help="How many timestamps to scan (deterministic sampling).",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Output dir for json/md report. Defaults to outputs/qa.",
    )
    return p.parse_args()


def _to_markdown(report: dict) -> str:
    s = report.get("summary", {})
    lines = [
        "# Source Metadata Validation Report",
        "",
        f"- status: `{s.get('status', 'unknown')}`",
        f"- sampled_count: `{s.get('sampled_count', 0)}`",
        f"- ok_count: `{s.get('ok_count', 0)}`",
        f"- missing_count: `{s.get('missing_count', 0)}`",
        f"- coverage: `{s.get('coverage', 0.0)}`",
        "",
        "## Distribution",
        f"- by_source: `{json.dumps(report.get('distribution', {}).get('by_source', {}), ensure_ascii=False)}`",
        f"- by_product_id: `{json.dumps(report.get('distribution', {}).get('by_product_id', {}), ensure_ascii=False)}`",
        "",
        "## Missing Samples (top 20)",
    ]
    misses = [c for c in report.get("checks", []) if not bool(c.get("ok", False))]
    for row in misses[:20]:
        lines.append(f"- `{json.dumps(row, ensure_ascii=False)}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    report = build_source_metadata_report(settings=settings, sample_limit=max(1, int(args.sample_limit)))

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (settings.outputs_root / "qa")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"source_metadata_{stamp}.json"
    md_path = out_dir / f"source_metadata_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    print(f"json={json_path}")
    print(f"md={md_path}")
    print(f"status={report.get('summary', {}).get('status', 'unknown')}")

    status = str(report.get("summary", {}).get("status", "unknown")).lower()
    if status == "fail":
        raise SystemExit(2)
    if status == "warn":
        # Non-blocking warning by design.
        raise SystemExit(1)


if __name__ == "__main__":
    main()
