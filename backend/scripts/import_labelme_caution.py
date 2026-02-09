from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.preprocess.labelme_io import labelme_json_to_binary_mask, save_binary_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import Labelme JSON polygons into caution_mask.png/.npy")
    p.add_argument(
        "--annotation-root",
        default="",
        help="Root path of annotation folders. Defaults to data/processed/annotation_pack",
    )
    p.add_argument(
        "--timestamp",
        default="",
        help="Optional single timestamp directory, e.g. 2024-10-31_18",
    )
    p.add_argument(
        "--json-name",
        default="quicklook.json",
        help="Labelme json filename inside each timestamp folder.",
    )
    p.add_argument("--label", default="caution", help="Label name to extract from shapes.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    root = (
        Path(args.annotation_root) if args.annotation_root else (settings.data_root / "processed" / "annotation_pack")
    )
    if not root.exists():
        raise FileNotFoundError(f"annotation root not found: {root}")

    if args.timestamp:
        folders = [root / args.timestamp]
    else:
        folders = [p for p in sorted(root.iterdir()) if p.is_dir() and len(p.name) == 13]

    converted = 0
    skipped = 0
    for folder in folders:
        json_path = folder / args.json_name
        if not json_path.exists():
            skipped += 1
            continue
        mask = labelme_json_to_binary_mask(json_path, target_label=args.label)
        save_binary_mask(mask, folder / "caution_mask.png", folder / "caution_mask.npy")
        converted += 1

    print(f"converted={converted} skipped={skipped} root={root}")


if __name__ == "__main__":
    main()
