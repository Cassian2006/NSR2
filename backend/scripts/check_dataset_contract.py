from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings


TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}$")
PRED_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}_\d{2})(?:_uncertainty)?\.npy$")


@dataclass
class Check:
    name: str
    status: str  # pass|warn|fail
    detail: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check NSR dataset contract consistency.")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="How many timestamps to sample for shape/dtype validation (default: 20).",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Report output directory. Defaults to outputs/qa",
    )
    return parser.parse_args()


def _select_sampled_timestamps(timestamps: list[str], limit: int) -> list[str]:
    if len(timestamps) <= limit:
        return timestamps
    idx = np.linspace(0, len(timestamps) - 1, limit, dtype=int)
    picked = [timestamps[i] for i in idx.tolist()]
    # Keep order and uniqueness.
    seen: set[str] = set()
    deduped: list[str] = []
    for ts in picked:
        if ts not in seen:
            seen.add(ts)
            deduped.append(ts)
    return deduped


def _check_timestamp_dirs(annotation_root: Path) -> tuple[list[str], Check]:
    if not annotation_root.exists():
        return [], Check(
            name="annotation_pack_root_exists",
            status="fail",
            detail={"path": str(annotation_root), "reason": "missing"},
        )
    all_dirs = sorted([p for p in annotation_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    valid = [p.name for p in all_dirs if TIMESTAMP_RE.match(p.name)]
    invalid = [p.name for p in all_dirs if not TIMESTAMP_RE.match(p.name)]
    status = "pass" if not invalid else "warn"
    return valid, Check(
        name="timestamp_dir_naming",
        status=status,
        detail={
            "annotation_pack_root": str(annotation_root),
            "total_dirs": len(all_dirs),
            "valid_timestamp_dirs": len(valid),
            "invalid_dir_count": len(invalid),
            "invalid_examples": invalid[:10],
        },
    )


def _check_pred_file_naming(pred_root: Path) -> Check:
    model_dir = pred_root / "unet_v1"
    if not model_dir.exists():
        return Check(
            name="pred_file_naming",
            status="warn",
            detail={"path": str(model_dir), "reason": "missing_unet_v1_dir"},
        )
    files = sorted(model_dir.glob("*.npy"))
    invalid = [p.name for p in files if not PRED_RE.match(p.name)]
    status = "pass" if not invalid else "warn"
    return Check(
        name="pred_file_naming",
        status=status,
        detail={
            "path": str(model_dir),
            "file_count": len(files),
            "invalid_count": len(invalid),
            "invalid_examples": invalid[:10],
        },
    )


def _check_ais_heatmap_file_naming(ais_heatmap_root: Path) -> Check:
    if not ais_heatmap_root.exists():
        return Check(
            name="ais_heatmap_file_naming",
            status="warn",
            detail={"path": str(ais_heatmap_root), "reason": "missing_root"},
        )
    files = sorted(ais_heatmap_root.rglob("*.npy"))
    invalid = [p.name for p in files if not TIMESTAMP_RE.match(p.stem)]
    status = "pass" if not invalid else "warn"
    return Check(
        name="ais_heatmap_file_naming",
        status=status,
        detail={
            "path": str(ais_heatmap_root),
            "file_count": len(files),
            "invalid_count": len(invalid),
            "invalid_examples": invalid[:10],
        },
    )


def _check_sample_shape_dtype(annotation_root: Path, sampled_timestamps: list[str]) -> tuple[Check, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    required = {"x_stack.npy", "blocked_mask.npy"}

    x_dtypes: set[str] = set()
    blocked_dtypes: set[str] = set()
    x_channels: set[int] = set()
    hw_shapes: set[tuple[int, int]] = set()

    for ts in sampled_timestamps:
        folder = annotation_root / ts
        file_names = {p.name for p in folder.glob("*.npy")}
        missing = sorted(required - file_names)

        row: dict[str, Any] = {"timestamp": ts, "missing_required": missing}
        if missing:
            failures.append({"timestamp": ts, "reason": "missing_required", "missing": missing})
            rows.append(row)
            continue

        x = np.load(folder / "x_stack.npy", mmap_mode="r")
        blocked = np.load(folder / "blocked_mask.npy", mmap_mode="r")
        row["x_stack_shape"] = list(x.shape)
        row["blocked_shape"] = list(blocked.shape)
        row["x_stack_dtype"] = str(x.dtype)
        row["blocked_dtype"] = str(blocked.dtype)

        x_dtypes.add(str(x.dtype))
        blocked_dtypes.add(str(blocked.dtype))
        if x.ndim == 3:
            x_channels.add(int(x.shape[0]))
            hw_shapes.add((int(x.shape[1]), int(x.shape[2])))
        if blocked.ndim == 2:
            hw_shapes.add((int(blocked.shape[0]), int(blocked.shape[1])))

        if x.ndim != 3:
            failures.append({"timestamp": ts, "reason": "x_stack_not_3d", "shape": list(x.shape)})
        if blocked.ndim != 2:
            failures.append({"timestamp": ts, "reason": "blocked_not_2d", "shape": list(blocked.shape)})
        if x.ndim == 3 and blocked.ndim == 2 and x.shape[1:] != blocked.shape:
            failures.append(
                {
                    "timestamp": ts,
                    "reason": "shape_mismatch",
                    "x_stack_hw": list(x.shape[1:]),
                    "blocked_hw": list(blocked.shape),
                }
            )

        rows.append(row)

    if failures:
        status = "fail"
    elif len(x_dtypes) > 1 or len(blocked_dtypes) > 1 or len(x_channels) > 1 or len(hw_shapes) > 1:
        status = "warn"
    else:
        status = "pass"

    check = Check(
        name="sample_shape_dtype_consistency",
        status=status,
        detail={
            "sampled_count": len(sampled_timestamps),
            "x_stack_dtypes": sorted(x_dtypes),
            "blocked_dtypes": sorted(blocked_dtypes),
            "x_stack_channels": sorted(x_channels),
            "hw_shapes": [list(v) for v in sorted(hw_shapes)],
            "failure_count": len(failures),
            "failure_examples": failures[:10],
        },
    )
    return check, rows


def build_contract_report(*, sample_limit: int) -> dict[str, Any]:
    settings = get_settings()
    checks: list[Check] = []

    timestamps, ts_check = _check_timestamp_dirs(settings.annotation_pack_root)
    checks.append(ts_check)
    checks.append(_check_pred_file_naming(settings.pred_root))
    checks.append(_check_ais_heatmap_file_naming(settings.ais_heatmap_root))

    sampled = _select_sampled_timestamps(timestamps, max(1, int(sample_limit)))
    sample_check, sample_rows = _check_sample_shape_dtype(settings.annotation_pack_root, sampled)
    checks.append(sample_check)

    statuses = [c.status for c in checks]
    if any(s == "fail" for s in statuses):
        overall = "fail"
    elif any(s == "warn" for s in statuses):
        overall = "warn"
    else:
        overall = "pass"

    report = {
        "summary": {
            "status": overall,
            "timestamp_count": len(timestamps),
            "sample_limit": max(1, int(sample_limit)),
            "sampled_count": len(sampled),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        },
        "paths": {
            "data_root": str(settings.data_root),
            "annotation_pack_root": str(settings.annotation_pack_root),
            "ais_heatmap_root": str(settings.ais_heatmap_root),
            "pred_root": str(settings.pred_root),
        },
        "checks": [asdict(c) for c in checks],
        "sample_rows": sample_rows,
    }
    return report


def _to_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines: list[str] = []
    lines.append("# Dataset Contract Report")
    lines.append("")
    lines.append(f"- status: `{summary.get('status', 'unknown')}`")
    lines.append(f"- timestamp_count: `{summary.get('timestamp_count', 0)}`")
    lines.append(f"- sampled_count: `{summary.get('sampled_count', 0)}`")
    lines.append(f"- sample_limit: `{summary.get('sample_limit', 0)}`")
    lines.append("")
    lines.append("## Checks")
    for c in report.get("checks", []):
        lines.append(f"- [{str(c.get('status', '')).upper()}] `{c.get('name', 'unknown')}`")
        lines.append(f"  detail: `{json.dumps(c.get('detail', {}), ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Sample Rows (first 20)")
    for row in report.get("sample_rows", [])[:20]:
        lines.append(f"- `{json.dumps(row, ensure_ascii=False)}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_contract_report(sample_limit=max(1, int(args.sample_limit)))
    settings = get_settings()

    out_dir = Path(args.out_dir) if args.out_dir else (settings.outputs_root / "qa")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"dataset_contract_{stamp}.json"
    md_path = out_dir / f"dataset_contract_{stamp}.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    print(f"json={json_path}")
    print(f"md={md_path}")
    print(f"status={report.get('summary', {}).get('status', 'unknown')}")

    status = str(report.get("summary", {}).get("status", "unknown")).lower()
    if status == "fail":
        raise SystemExit(2)
    if status == "warn":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

