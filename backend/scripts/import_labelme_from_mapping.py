from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.preprocess.labelme_io import labelme_json_to_binary_mask, save_binary_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Import Labelme JSONs from renamed set via mapping.csv into annotation_pack caution masks"
    )
    p.add_argument(
        "--source-dir",
        default="",
        help="Directory containing renamed PNG/JSON files and mapping.csv. Defaults to labelme_blocked_50_enhanced.",
    )
    p.add_argument(
        "--mapping-csv",
        default="mapping.csv",
        help="Mapping CSV filename inside source-dir.",
    )
    p.add_argument(
        "--json-suffix",
        default="_riskhint.json",
        help="JSON suffix appended to blocked_{idx:03d}. Default: _riskhint.json",
    )
    p.add_argument(
        "--name-template",
        default="blocked_{idx:03d}{suffix}",
        help="Template to build JSON filename from index and suffix. Fields: {idx}, {suffix}.",
    )
    p.add_argument(
        "--label",
        default="caution",
        help="Label name to extract from labelme JSONs.",
    )
    p.add_argument(
        "--target-root",
        default="",
        help="Target annotation root with timestamp folders. Defaults to data/processed/annotation_pack",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    source_dir = (
        Path(args.source_dir)
        if args.source_dir
        else (settings.data_root / "processed" / "annotation_pack" / "labelme_blocked_50_enhanced")
    )
    mapping_path = source_dir / args.mapping_csv
    target_root = (
        Path(args.target_root) if args.target_root else (settings.data_root / "processed" / "annotation_pack")
    )
    if not source_dir.exists():
        raise FileNotFoundError(f"source-dir not found: {source_dir}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"mapping csv not found: {mapping_path}")
    if not target_root.exists():
        raise FileNotFoundError(f"target-root not found: {target_root}")

    idx_re = re.compile(r"^\d+$")
    converted = 0
    missing_json = 0
    missing_target = 0

    with mapping_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_str = str(row.get("index", "")).strip()
            ts = str(row.get("timestamp", "")).strip()
            if not idx_re.match(idx_str) or not ts:
                continue
            idx = int(idx_str)

            json_name = args.name_template.format(idx=idx, suffix=args.json_suffix)
            json_path = source_dir / json_name
            if not json_path.exists():
                missing_json += 1
                continue

            target_dir = target_root / ts
            if not target_dir.exists():
                missing_target += 1
                continue

            mask = labelme_json_to_binary_mask(json_path, target_label=args.label)
            save_binary_mask(mask, target_dir / "caution_mask.png", target_dir / "caution_mask.npy")
            converted += 1

    print(f"source_dir={source_dir}")
    print(f"mapping={mapping_path}")
    print(f"target_root={target_root}")
    print(f"converted={converted} missing_json={missing_json} missing_target={missing_target}")


if __name__ == "__main__":
    main()
