from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.preprocess.labelme_io import labelme_json_to_binary_mask, save_binary_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Finalize active-learning review labels into annotation_pack caution masks"
    )
    p.add_argument(
        "--review-dir",
        default="",
        help="Review folder containing active_XXX_review.json and mapping_review.csv.",
    )
    p.add_argument(
        "--mapping-csv",
        default="mapping_review.csv",
        help="Mapping csv filename in review-dir.",
    )
    p.add_argument(
        "--target-root",
        default="",
        help="Target annotation_pack root. Defaults to data/processed/annotation_pack",
    )
    p.add_argument(
        "--merge-with-suggest",
        action="store_true",
        help="Merge human polygons with caution_suggested.npy (OR). Recommended for review workflow.",
    )
    p.add_argument(
        "--label",
        default="caution",
        help="Label name in Labelme JSON to read as caution mask.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    review_dir = (
        Path(args.review_dir)
        if args.review_dir
        else (
            settings.outputs_root
            / "active_learning"
            / "active_20260209_225559"
            / "labelme_active_topk"
            / "review_overlay20"
        )
    )
    mapping_path = review_dir / args.mapping_csv
    target_root = (
        Path(args.target_root) if args.target_root else (settings.data_root / "processed" / "annotation_pack")
    )

    if not review_dir.exists():
        raise FileNotFoundError(f"review-dir not found: {review_dir}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"mapping csv not found: {mapping_path}")
    if not target_root.exists():
        raise FileNotFoundError(f"target root not found: {target_root}")

    merged_count = 0
    suggest_only_count = 0
    human_only_count = 0
    missing_target = 0

    with mapping_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = int(str(row.get("rank", "0")).strip() or "0")
            ts = str(row.get("timestamp", "")).strip()
            if rank <= 0 or not ts:
                continue
            target_dir = target_root / ts
            if not target_dir.exists():
                missing_target += 1
                continue

            blocked_path = target_dir / "blocked_mask.npy"
            if not blocked_path.exists():
                missing_target += 1
                continue
            blocked = np.load(blocked_path)
            h, w = int(blocked.shape[0]), int(blocked.shape[1])

            suggest = np.zeros((h, w), dtype=np.uint8)
            if args.merge_with_suggest:
                suggest_path = target_dir / "caution_suggested.npy"
                if suggest_path.exists():
                    s = np.load(suggest_path)
                    if s.shape == (h, w):
                        suggest = (s > 0).astype(np.uint8)

            json_path = review_dir / f"active_{rank:03d}_review.json"
            human = np.zeros((h, w), dtype=np.uint8)
            if json_path.exists():
                hm = labelme_json_to_binary_mask(json_path, target_label=args.label)
                if hm.shape == (h, w):
                    human = (hm > 0).astype(np.uint8)
                else:
                    # size mismatch fallback: skip human for this sample
                    human = np.zeros((h, w), dtype=np.uint8)

            if int(human.sum()) > 0 and int(suggest.sum()) > 0:
                merged_count += 1
            elif int(human.sum()) > 0:
                human_only_count += 1
            elif int(suggest.sum()) > 0:
                suggest_only_count += 1

            caution = np.maximum(human, suggest).astype(np.uint8)
            save_binary_mask(caution, target_dir / "caution_mask.png", target_dir / "caution_mask.npy")

    print(f"review_dir={review_dir}")
    print(f"mapping={mapping_path}")
    print(f"target_root={target_root}")
    print(
        f"merged={merged_count} human_only={human_only_count} suggest_only={suggest_only_count} "
        f"missing_target={missing_target}"
    )


if __name__ == "__main__":
    main()
