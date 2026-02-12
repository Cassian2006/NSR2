from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    p.add_argument(
        "--export-batches",
        action="store_true",
        help="Export ready-to-label batches with riskhint/landmask/blocked overlay and mapping csv.",
    )
    p.add_argument("--batch-size", type=int, default=20, help="How many samples per exported batch.")
    p.add_argument(
        "--batches-root",
        default="",
        help="Batch export root. Defaults to <out-root>/label_batches",
    )
    p.add_argument(
        "--resume-batches",
        action="store_true",
        help="Continue exporting only not-yet-exported timestamps using state file under batches root.",
    )
    p.add_argument(
        "--only-unlabeled-batches",
        action="store_true",
        help="Export only samples whose caution_mask has no positive pixels.",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Optional cap of newly created batches (0 = no cap).",
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


def _normalize_on_finite(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    vals = x[finite]
    lo = float(np.percentile(vals, 5))
    hi = float(np.percentile(vals, 95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    out = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32)


def _render_landmask(blocked_mask: np.ndarray) -> np.ndarray:
    sea = blocked_mask == 0
    img = np.zeros(blocked_mask.shape, dtype=np.uint8)
    img[sea] = 255
    return img


def _render_blocked_overlay(base_rgb: np.ndarray, blocked_mask: np.ndarray) -> np.ndarray:
    out = base_rgb.astype(np.float32)
    blk = blocked_mask > 0
    out[blk, 0] = out[blk, 0] * 0.35 + 255.0 * 0.65
    out[blk, 1] = out[blk, 1] * 0.35
    out[blk, 2] = out[blk, 2] * 0.35
    return np.clip(out, 0, 255).astype(np.uint8)


def _render_riskhint(
    stack: np.ndarray,
    channel_names: list[str],
    blocked_mask: np.ndarray,
    base_rgb: np.ndarray,
) -> np.ndarray:
    def idx_of(name: str, fallback: int) -> int:
        if name in channel_names:
            return channel_names.index(name)
        return min(fallback, stack.shape[0] - 1)

    i_ice = idx_of("ice_conc", 0)
    i_wave = idx_of("wave_hs", 2)
    i_u10 = idx_of("wind_u10", 3)
    i_v10 = idx_of("wind_v10", 4)
    i_ais = idx_of("ais_heatmap", stack.shape[0] - 1)

    ice = _normalize_on_finite(stack[i_ice])
    wave = _normalize_on_finite(stack[i_wave])
    wind = _normalize_on_finite(np.sqrt(np.square(stack[i_u10]) + np.square(stack[i_v10])))
    ais = _normalize_on_finite(stack[i_ais])

    risk = 0.42 * ice + 0.28 * wave + 0.20 * wind + 0.10 * (1.0 - ais)
    risk = np.clip(risk, 0.0, 1.0).astype(np.float32)

    heat = np.zeros((*risk.shape, 3), dtype=np.uint8)
    heat[..., 0] = (risk * 255.0).astype(np.uint8)
    heat[..., 1] = ((1.0 - risk) * 190.0).astype(np.uint8)
    heat[..., 2] = ((0.5 + 0.5 * (1.0 - risk)) * 120.0).astype(np.uint8)

    out = (0.48 * base_rgb.astype(np.float32) + 0.52 * heat.astype(np.float32)).astype(np.uint8)
    # Keep blocked visually distinct.
    blk = blocked_mask > 0
    out[blk] = np.array([30, 30, 30], dtype=np.uint8)
    return out


def _is_unlabeled_caution(caution_npy: Path) -> bool:
    if not caution_npy.exists():
        return True
    try:
        mask = np.load(caution_npy, mmap_mode="r")
        return int((mask > 0).sum()) == 0
    except Exception:
        return True


def _iter_pack_timestamps(out_root: Path) -> list[str]:
    ts: list[str] = []
    if not out_root.exists():
        return ts
    for folder in sorted(out_root.iterdir()):
        if not folder.is_dir():
            continue
        if len(folder.name) != 13:
            continue
        if (folder / "meta.json").exists():
            ts.append(folder.name)
    return ts


def _load_batch_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"next_batch_index": 1, "exported_timestamps": []}
    try:
        obj = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return {"next_batch_index": 1, "exported_timestamps": []}
        return {
            "next_batch_index": int(obj.get("next_batch_index", 1)),
            "exported_timestamps": list(obj.get("exported_timestamps", [])),
        }
    except Exception:
        return {"next_batch_index": 1, "exported_timestamps": []}


def _save_batch_state(state_path: Path, *, next_batch_index: int, exported_timestamps: list[str]) -> None:
    payload = {
        "next_batch_index": int(max(1, next_batch_index)),
        "exported_timestamps": sorted(set(str(v) for v in exported_timestamps)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_batch_readme(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Label Batch Guide",
                "",
                "- Label on *_riskhint.png (polygon label: caution).",
                "- *_landmask.png and *_blocked_overlay.png are reference layers.",
                "- Use mapping.csv to map sample file -> timestamp.",
            ]
        ),
        encoding="utf-8",
    )


def _export_label_batches(
    *,
    out_root: Path,
    batches_root: Path,
    batch_size: int,
    resume_batches: bool,
    only_unlabeled: bool,
    max_batches: int,
) -> dict[str, Any]:
    timestamps = _iter_pack_timestamps(out_root)
    if not timestamps:
        return {"created_batches": 0, "exported": 0, "candidate_count": 0, "batches_root": str(batches_root)}

    state_path = batches_root / "batch_state.json"
    state = _load_batch_state(state_path)
    exported_set = set(str(v) for v in state.get("exported_timestamps", []))

    candidates: list[str] = []
    for ts in timestamps:
        src = out_root / ts
        needed = [
            src / "quicklook_riskhint.png",
            src / "quicklook_landmask.png",
            src / "quicklook_blocked_overlay.png",
        ]
        if not all(p.exists() for p in needed):
            continue
        if only_unlabeled and (not _is_unlabeled_caution(src / "caution_mask.npy")):
            continue
        if resume_batches and ts in exported_set:
            continue
        candidates.append(ts)

    if not candidates:
        return {
            "created_batches": 0,
            "exported": 0,
            "candidate_count": 0,
            "batches_root": str(batches_root),
            "message": "no candidates to export",
        }

    batch_size = max(1, int(batch_size))
    max_batches = max(0, int(max_batches))
    next_idx = int(state.get("next_batch_index", 1))
    created_batches = 0
    exported_now = 0
    global_rows: list[dict[str, Any]] = []

    for i in range(0, len(candidates), batch_size):
        if max_batches > 0 and created_batches >= max_batches:
            break
        chunk = candidates[i : i + batch_size]
        batch_dir = batches_root / f"batch_{next_idx:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        _write_batch_readme(batch_dir / "README.txt")

        rows: list[dict[str, Any]] = []
        for local_i, ts in enumerate(chunk, start=1):
            src = out_root / ts
            base = f"sample_{local_i:03d}"
            risk_name = f"{base}_riskhint.png"
            land_name = f"{base}_landmask.png"
            blocked_name = f"{base}_blocked_overlay.png"
            shutil.copy2(src / "quicklook_riskhint.png", batch_dir / risk_name)
            shutil.copy2(src / "quicklook_landmask.png", batch_dir / land_name)
            shutil.copy2(src / "quicklook_blocked_overlay.png", batch_dir / blocked_name)

            row = {
                "batch_id": f"batch_{next_idx:03d}",
                "index_in_batch": local_i,
                "timestamp": ts,
                "riskhint_file": risk_name,
                "landmask_file": land_name,
                "blocked_overlay_file": blocked_name,
                "source_dir": str(src),
            }
            rows.append(row)
            global_rows.append(row)
            exported_set.add(ts)
            exported_now += 1

        mapping_path = batch_dir / "mapping.csv"
        import csv

        with mapping_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "batch_id",
                    "index_in_batch",
                    "timestamp",
                    "riskhint_file",
                    "landmask_file",
                    "blocked_overlay_file",
                    "source_dir",
                ],
            )
            w.writeheader()
            w.writerows(rows)

        batch_meta = {
            "batch_id": f"batch_{next_idx:03d}",
            "size": len(rows),
            "mapping_csv": str(mapping_path),
        }
        (batch_dir / "meta.json").write_text(json.dumps(batch_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        created_batches += 1
        next_idx += 1

    if global_rows:
        import csv

        index_csv = batches_root / "batches_index.csv"
        batches_root.mkdir(parents=True, exist_ok=True)
        with index_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "batch_id",
                    "index_in_batch",
                    "timestamp",
                    "riskhint_file",
                    "landmask_file",
                    "blocked_overlay_file",
                    "source_dir",
                ],
            )
            w.writeheader()
            w.writerows(global_rows)

    _save_batch_state(
        state_path,
        next_batch_index=next_idx,
        exported_timestamps=sorted(exported_set),
    )

    return {
        "created_batches": created_batches,
        "exported": exported_now,
        "candidate_count": len(candidates),
        "batches_root": str(batches_root),
    }


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
            landmask = _render_landmask(blocked_mask)
            _write_mask_png(landmask, out_dir / "quicklook_landmask.png")
            blocked_overlay = _render_blocked_overlay(rgb, blocked_mask)
            _write_rgb_png(blocked_overlay, out_dir / "quicklook_blocked_overlay.png")
            riskhint = _render_riskhint(bundle.stack, bundle.channel_names, blocked_mask, rgb)
            _write_rgb_png(riskhint, out_dir / "quicklook_riskhint.png")

        meta = {
            "timestamp": ts,
            "x_stack": "x_stack.npy",
            "blocked_mask": "blocked_mask.npy",
            "caution_mask": "caution_mask.npy",
            "riskhint": "quicklook_riskhint.png",
            "landmask": "quicklook_landmask.png",
            "blocked_overlay": "quicklook_blocked_overlay.png",
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

    if args.export_batches:
        batches_root = Path(args.batches_root) if args.batches_root else (out_root / "label_batches")
        batch_summary = _export_label_batches(
            out_root=out_root,
            batches_root=batches_root,
            batch_size=args.batch_size,
            resume_batches=args.resume_batches,
            only_unlabeled=args.only_unlabeled_batches,
            max_batches=args.max_batches,
        )
        print(
            "batch_export="
            + json.dumps(
                batch_summary,
                ensure_ascii=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
