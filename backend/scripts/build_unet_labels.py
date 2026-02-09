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

from app.core.config import get_settings
from app.preprocess.unet_dataset import merge_multiclass_label


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge manual caution + blocked mask into U-Net class labels and manifest"
    )
    p.add_argument(
        "--annotation-root",
        default="",
        help="Annotation pack root. Defaults to data/processed/annotation_pack",
    )
    p.add_argument(
        "--manifest-path",
        default="",
        help="Output manifest CSV. Defaults to data/processed/unet_manifest.csv",
    )
    p.add_argument(
        "--val-months",
        default="202410",
        help="Comma-separated validation months in YYYYMM. Default: 202410",
    )
    p.add_argument(
        "--skip-empty-caution",
        action="store_true",
        help="Skip samples where caution mask has no positive pixels.",
    )
    return p.parse_args()


def _load_caution_mask(folder: Path) -> np.ndarray:
    png_path = folder / "caution_mask.png"
    npy_path = folder / "caution_mask.npy"

    if png_path.exists():
        try:
            from PIL import Image

            img = Image.open(png_path).convert("L")
            arr = np.array(img)
            return (arr >= 127).astype(np.uint8)
        except Exception:
            pass

    if npy_path.exists():
        arr = np.load(npy_path)
        if arr.ndim != 2:
            raise ValueError(f"caution_mask.npy must be 2D: {npy_path}")
        return (arr > 0).astype(np.uint8)

    raise FileNotFoundError(f"Missing caution mask in {folder}")


def main() -> None:
    args = parse_args()
    settings = get_settings()

    annotation_root = (
        Path(args.annotation_root) if args.annotation_root else (settings.data_root / "processed" / "annotation_pack")
    )
    manifest_path = (
        Path(args.manifest_path) if args.manifest_path else (settings.data_root / "processed" / "unet_manifest.csv")
    )
    val_months = {m.strip() for m in args.val_months.split(",") if m.strip()}

    if not annotation_root.exists():
        raise FileNotFoundError(f"Annotation root not found: {annotation_root}")

    manifest_rows: list[dict[str, str]] = []
    skipped = 0
    for folder in sorted(annotation_root.iterdir()):
        if not folder.is_dir():
            continue
        ts = folder.name
        if len(ts) != 13:
            continue

        x_path = folder / "x_stack.npy"
        blocked_path = folder / "blocked_mask.npy"
        if not x_path.exists() or not blocked_path.exists():
            skipped += 1
            continue

        blocked = np.load(blocked_path)
        caution = _load_caution_mask(folder)
        if blocked.shape != caution.shape:
            raise ValueError(f"Mask shape mismatch in {folder}: blocked={blocked.shape}, caution={caution.shape}")
        if args.skip_empty_caution and int(caution.sum()) == 0:
            skipped += 1
            continue

        y = merge_multiclass_label(blocked, caution)
        y_path = folder / "y_class.npy"
        np.save(y_path, y.astype(np.uint8))

        month = ts[:7].replace("-", "")
        split = "val" if month in val_months else "train"
        manifest_rows.append(
            {
                "timestamp": ts,
                "split": split,
                "x_path": str(x_path),
                "y_path": str(y_path),
                "blocked_path": str(blocked_path),
                "caution_path": str(folder / "caution_mask.png"),
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "split", "x_path", "y_path", "blocked_path", "caution_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "annotation_root": str(annotation_root),
        "manifest_path": str(manifest_path),
        "rows": len(manifest_rows),
        "train_rows": sum(1 for r in manifest_rows if r["split"] == "train"),
        "val_rows": sum(1 for r in manifest_rows if r["split"] == "val"),
        "skipped": skipped,
        "val_months": sorted(val_months),
    }
    summary_path = manifest_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")
    print(f"rows={summary['rows']} train={summary['train_rows']} val={summary['val_rows']} skipped={skipped}")


if __name__ == "__main__":
    main()
