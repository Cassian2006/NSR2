from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.preprocess.unet_dataset import (
    list_sample_timestamps,
    load_feature_stack,
    make_blocked_mask_from_bathy,
    quicklook_rgb,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare manual annotation pack for U-Net (x_stack + caution template + blocked mask)"
    )
    p.add_argument(
        "--months",
        default="",
        help="Comma-separated months in YYYYMM, e.g. 202407,202408. Empty means all available months.",
    )
    p.add_argument("--heatmap-tag", default="7d", help="Subfolder under data/ais_heatmap")
    p.add_argument(
        "--blocked-if-bathy-gte",
        type=float,
        default=0.0,
        help="Mark blocked where bathy >= threshold (and bathy NaN).",
    )
    p.add_argument("--limit", type=int, default=0, help="Optional max sample count for dry runs.")
    p.add_argument("--overwrite-stack", action="store_true", help="Overwrite existing x_stack.npy")
    p.add_argument(
        "--reset-caution",
        action="store_true",
        help="Reset caution templates to all-zero (both .npy and .png).",
    )
    p.add_argument("--skip-quicklook", action="store_true", help="Do not generate quicklook PNG.")
    p.add_argument(
        "--out-root",
        default="",
        help="Output directory. Defaults to data/processed/annotation_pack",
    )
    return p.parse_args()


def _write_mask_png(mask: np.ndarray, path: Path) -> bool:
    try:
        from PIL import Image
    except Exception:
        return False
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return True


def _write_rgb_png(rgb: np.ndarray, path: Path) -> bool:
    try:
        from PIL import Image
    except Exception:
        return False
    img = Image.fromarray(rgb)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return True


def _write_guide_once(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# U-Net Annotation Pack Guide",
                "",
                "Per timestamp folder contains:",
                "- `x_stack.npy`: model input stack `(C,H,W)`",
                "- `blocked_mask.npy`: auto-generated blocked area from bathy",
                "- `caution_mask.npy`: your editable caution mask (0/1)",
                "- `caution_mask.png`: same caution mask for image editing",
                "- `quicklook.png`: RGB helper view (R=ice, G=wave, B=AIS heatmap)",
                "",
                "Manual annotation workflow:",
                "1. Edit `caution_mask.png` in any image tool. White=CAUTION, Black=non-caution.",
                "2. Keep image size unchanged.",
                "3. Run `python backend/scripts/build_unet_labels.py` to merge labels and create manifest.",
                "",
                "Label classes:",
                "- 0 = SAFE",
                "- 1 = CAUTION (from your manual mask)",
                "- 2 = BLOCKED (from bathy mask; overrides caution)",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    settings = get_settings()

    samples_root = settings.processed_samples_root
    env_root = settings.data_root / "interim" / "env_grids"
    heatmap_root = settings.ais_heatmap_root / args.heatmap_tag
    out_root = Path(args.out_root) if args.out_root else (settings.data_root / "processed" / "annotation_pack")

    if not env_root.exists():
        raise FileNotFoundError(f"Missing env root: {env_root}")
    if not heatmap_root.exists():
        raise FileNotFoundError(f"Missing heatmap root: {heatmap_root}")

    months = {m.strip() for m in args.months.split(",") if m.strip()} or None
    timestamps = list_sample_timestamps(samples_root, months=months)
    if args.limit > 0:
        timestamps = timestamps[: args.limit]

    _write_guide_once(out_root / "README.md")

    ok_count = 0
    skipped: list[str] = []
    for ts in timestamps:
        try:
            bundle = load_feature_stack(
                timestamp=ts,
                env_root=env_root,
                heatmap_root=heatmap_root,
            )
        except Exception:
            skipped.append(ts)
            continue

        out_dir = out_root / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        x_stack_path = out_dir / "x_stack.npy"
        if args.overwrite_stack or (not x_stack_path.exists()):
            np.save(x_stack_path, bundle.stack.astype(np.float32))

        bathy_idx = bundle.channel_names.index("bathy")
        bathy = bundle.stack[bathy_idx]
        blocked_mask = make_blocked_mask_from_bathy(
            bathy,
            blocked_if_bathy_gte=args.blocked_if_bathy_gte,
        )
        np.save(out_dir / "blocked_mask.npy", blocked_mask.astype(np.uint8))

        caution_npy = out_dir / "caution_mask.npy"
        caution_png = out_dir / "caution_mask.png"
        if args.reset_caution or (not caution_npy.exists()):
            caution = np.zeros(blocked_mask.shape, dtype=np.uint8)
            np.save(caution_npy, caution)
            _write_mask_png(caution, caution_png)
        elif not caution_png.exists():
            caution = np.load(caution_npy)
            _write_mask_png(caution, caution_png)

        if not args.skip_quicklook:
            rgb = quicklook_rgb(bundle.stack, bundle.channel_names)
            _write_rgb_png(rgb, out_dir / "quicklook.png")

        meta = {
            "timestamp": ts,
            "x_stack": "x_stack.npy",
            "blocked_mask": "blocked_mask.npy",
            "caution_mask": "caution_mask.npy",
            "channel_names": bundle.channel_names,
            "shape": [int(bundle.stack.shape[1]), int(bundle.stack.shape[2])],
            "blocked_rule": {
                "type": "bathy_gte_or_nan",
                "threshold": args.blocked_if_bathy_gte,
            },
            "has_bathy": bool(bundle.has_bathy),
        }
        with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        ok_count += 1

    print(f"prepared={ok_count} out_root={out_root}")
    if skipped:
        print(f"skipped={len(skipped)}")
        print("first_skipped=" + ",".join(skipped[:8]))


if __name__ == "__main__":
    main()
