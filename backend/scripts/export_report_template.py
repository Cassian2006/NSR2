from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.core.compliance import build_compliance_notices
from app.core.config import get_settings
from app.core.gallery import GalleryService
from app.core.report_template import build_report_template, report_template_to_csv, report_template_to_markdown
from app.core.risk_report import build_risk_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export standardized report template from a gallery run.")
    parser.add_argument("--gallery-id", required=True, help="Gallery run id")
    parser.add_argument(
        "--formats",
        default="json,csv,md",
        help="Comma-separated formats, subset of: json,csv,md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    service = GalleryService()
    item = service.get_item(args.gallery_id)
    if item is None:
        raise SystemExit(f"gallery item not found: {args.gallery_id}")

    compliance = build_compliance_notices(
        settings=settings,
        context="export",
        timestamp=str(item.get("timestamp", "")),
    )
    risk_report = build_risk_report(item)
    report = build_report_template(gallery_item=item, risk_report=risk_report, compliance=compliance)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = settings.outputs_root / "release"
    out_root.mkdir(parents=True, exist_ok=True)
    requested_formats = {f.strip().lower() for f in str(args.formats).split(",") if f.strip()}
    outputs: dict[str, str] = {}

    if "json" in requested_formats:
        path = out_root / f"report_template_v1_{args.gallery_id}_{ts}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs["json"] = str(path)
    if "csv" in requested_formats:
        path = out_root / f"report_template_v1_{args.gallery_id}_{ts}.csv"
        path.write_text(report_template_to_csv(report), encoding="utf-8")
        outputs["csv"] = str(path)
    if "md" in requested_formats or "markdown" in requested_formats:
        path = out_root / f"report_template_v1_{args.gallery_id}_{ts}.md"
        path.write_text(report_template_to_markdown(report), encoding="utf-8")
        outputs["md"] = str(path)

    print(json.dumps({"status": "ok", "gallery_id": args.gallery_id, "outputs": outputs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
