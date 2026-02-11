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
from app.core.data_quality import build_data_quality_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate dataset quality report (json + markdown).")
    p.add_argument("--sample-limit", type=int, default=120)
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Defaults to outputs/qa",
    )
    return p.parse_args()


def _to_markdown(report: dict) -> str:
    lines: list[str] = []
    summary = report.get("summary", {})
    lines.append("# Dataset Quality Report")
    lines.append("")
    lines.append(f"- status: `{summary.get('status', 'unknown')}`")
    lines.append(f"- timestamp_count: `{summary.get('timestamp_count', 0)}`")
    lines.append(f"- range: `{summary.get('first_timestamp', '-')}` -> `{summary.get('last_timestamp', '-')}`")
    lines.append(f"- issues_count: `{summary.get('issues_count', 0)}`")
    lines.append("")
    lines.append("## Checks")
    for c in report.get("checks", []):
        status = str(c.get("status", "unknown")).upper()
        name = str(c.get("name", "unknown"))
        detail = c.get("detail", {})
        lines.append(f"- [{status}] `{name}`")
        lines.append(f"  detail: `{json.dumps(detail, ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Issues")
    for issue in report.get("issues", []):
        lines.append(f"- `{issue}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    report = build_data_quality_report(settings=settings, sample_limit=max(8, min(500, int(args.sample_limit))))
    out_dir = Path(args.out_dir) if args.out_dir else (settings.outputs_root / "qa")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"dataset_quality_{stamp}.json"
    md_path = out_dir / f"dataset_quality_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")
    print(f"json={json_path}")
    print(f"md={md_path}")
    print(f"status={report.get('summary', {}).get('status', 'unknown')}")


if __name__ == "__main__":
    main()
