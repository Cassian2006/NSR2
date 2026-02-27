from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.dataset import get_dataset_service
from app.core.timestamps_index import build_timestamps_index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build standardized timestamps index with gap report.")
    p.add_argument(
        "--month",
        default="all",
        help="Month filter like 2024-07. Default: all.",
    )
    p.add_argument(
        "--step-hours",
        type=int,
        default=1,
        help="Target timeline step hours, e.g. 1 or 3.",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output path. Defaults to data/processed/timestamps_index.json.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    service = get_dataset_service()
    month = str(args.month).strip()
    if not month:
        month = "all"

    source = service.list_source_timestamps(month=month)
    report = build_timestamps_index(
        source_timestamps=source,
        step_hours=max(1, int(args.step_hours)),
        source_kind="annotation_pack",
    )
    report["filter_month"] = month

    out_path = Path(args.out).resolve() if args.out else settings.timestamps_index_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report.get("summary", {})
    print(f"out={out_path}")
    print(
        "summary="
        + json.dumps(
            {
                "status": summary.get("status", "unknown"),
                "step_hours": summary.get("step_hours", 0),
                "source_count": summary.get("source_count", 0),
                "expected_count": summary.get("expected_count", 0),
                "available_count": summary.get("available_count", 0),
                "missing_count": summary.get("missing_count", 0),
                "coverage": summary.get("coverage", 0.0),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
