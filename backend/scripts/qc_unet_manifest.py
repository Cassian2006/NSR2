from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model.train_quality import evaluate_sample_quality


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run quality checks for U-Net manifest samples.")
    p.add_argument(
        "--manifest",
        default="",
        help="Input manifest CSV. Defaults to data/processed/unet_manifest_labeled.csv",
    )
    p.add_argument(
        "--min-foreground-ratio",
        type=float,
        default=2e-4,
        help="Minimum positive ratio (caution+blocked).",
    )
    p.add_argument(
        "--max-nan-ratio",
        type=float,
        default=0.2,
        help="Maximum allowed non-finite ratio in x_stack.",
    )
    p.add_argument(
        "--out-json",
        default="",
        help="Output JSON report path. Defaults to data/processed/unet_manifest_quality_report.json",
    )
    return p.parse_args()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data"
    manifest = Path(args.manifest) if args.manifest else (data_root / "processed" / "unet_manifest_labeled.csv")
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    rows = _load_rows(manifest)
    bad: list[dict] = []
    ok = 0

    for row in rows:
        x_path = Path(str(row.get("x_path", "")).strip())
        y_path = Path(str(row.get("y_path", "")).strip())
        if not x_path.exists() or not y_path.exists():
            bad.append(
                {
                    "timestamp": str(row.get("timestamp", "")),
                    "split": str(row.get("split", "")),
                    "x_path": str(x_path),
                    "y_path": str(y_path),
                    "reasons": ["missing_file"],
                    "stats": {},
                }
            )
            continue
        try:
            x = np.load(x_path, mmap_mode="r")
            y = np.load(y_path, mmap_mode="r")
            qc = evaluate_sample_quality(
                np.asarray(x),
                np.asarray(y),
                min_foreground_ratio=args.min_foreground_ratio,
                max_nan_ratio=args.max_nan_ratio,
            )
        except Exception as exc:
            bad.append(
                {
                    "timestamp": str(row.get("timestamp", "")),
                    "split": str(row.get("split", "")),
                    "x_path": str(x_path),
                    "y_path": str(y_path),
                    "reasons": [f"load_error:{exc}"],
                    "stats": {},
                }
            )
            continue

        if qc.ok:
            ok += 1
        else:
            bad.append(
                {
                    "timestamp": str(row.get("timestamp", "")),
                    "split": str(row.get("split", "")),
                    "x_path": str(x_path),
                    "y_path": str(y_path),
                    "reasons": qc.reasons,
                    "stats": qc.stats,
                }
            )

    report = {
        "manifest": str(manifest),
        "rows": len(rows),
        "ok_rows": ok,
        "bad_rows": len(bad),
        "min_foreground_ratio": float(args.min_foreground_ratio),
        "max_nan_ratio": float(args.max_nan_ratio),
        "bad_samples": bad,
    }
    out = (
        Path(args.out_json)
        if args.out_json
        else (data_root / "processed" / "unet_manifest_quality_report.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest={manifest}")
    print(f"rows={len(rows)} ok={ok} bad={len(bad)}")
    print(f"report={out}")


if __name__ == "__main__":
    main()
