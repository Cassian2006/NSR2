from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.dataset import DatasetService
from app.preprocess.ais_heatmap import GridSpec, to_cell


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate demo-grade AIS heatmaps from cleaned AIS CSV files")
    p.add_argument("--tag", default="demo_v1", help="Output subfolder under data/ais_heatmap/")
    p.add_argument("--lat-min", type=float, default=60.0)
    p.add_argument("--lat-max", type=float, default=80.0)
    p.add_argument("--lon-min", type=float, default=20.0)
    p.add_argument("--lon-max", type=float, default=180.0)
    p.add_argument("--months", nargs="*", default=["202407", "202408", "202409", "202410"])
    return p.parse_args()


def box_blur3x3(field: np.ndarray, passes: int = 1) -> np.ndarray:
    out = field.astype(np.float32)
    for _ in range(max(0, int(passes))):
        pad = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            pad[:-2, :-2]
            + pad[:-2, 1:-1]
            + pad[:-2, 2:]
            + pad[1:-1, :-2]
            + pad[1:-1, 1:-1]
            + pad[1:-1, 2:]
            + pad[2:, :-2]
            + pad[2:, 1:-1]
            + pad[2:, 2:]
        ) / 9.0
    return out.astype(np.float32)


def normalize01(arr: np.ndarray) -> np.ndarray:
    out = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vmax = float(np.max(out)) if out.size else 0.0
    if vmax > 1e-6:
        out /= vmax
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def load_histogram(csv_path: Path, grid: GridSpec) -> np.ndarray:
    hist = np.zeros((grid.height, grid.width), dtype=np.float32)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        header = f.readline().strip().split(",")
        lon_idx = header.index("lon")
        lat_idx = header.index("lat")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= max(lon_idx, lat_idx):
                continue
            try:
                lon = float(parts[lon_idx])
                lat = float(parts[lat_idx])
            except ValueError:
                continue
            cell = to_cell(lat, lon, grid)
            if cell is None:
                continue
            hist[cell] += 1.0
    return hist


def build_demo_heatmap(month_hist: np.ndarray, season_prior: np.ndarray) -> np.ndarray:
    local = normalize01(np.log1p(month_hist))
    local = normalize01(box_blur3x3(local, passes=2))
    season = normalize01(box_blur3x3(season_prior, passes=3))
    blended = normalize01(local * 0.72 + season * 0.28)
    q90 = float(np.quantile(blended[blended > 1e-8], 0.90)) if np.any(blended > 1e-8) else 1.0
    if q90 > 1e-6:
        blended = np.clip(blended / q90, 0.0, 1.0)
    return blended.astype(np.float32)


def main() -> None:
    args = parse_args()
    settings = get_settings()
    dataset = DatasetService()

    sample_ts = "2024-07-01_00"
    x_stack = np.load(settings.annotation_pack_root / sample_ts / "x_stack.npy", mmap_mode="r")
    h, w = int(x_stack.shape[1]), int(x_stack.shape[2])
    grid = GridSpec(
        height=h,
        width=w,
        lat_min=float(args.lat_min),
        lat_max=float(args.lat_max),
        lon_min=float(args.lon_min),
        lon_max=float(args.lon_max),
    )

    month_hists: dict[str, np.ndarray] = {}
    for month in args.months:
        csv_path = settings.data_root / "processed" / "ais_cleaned" / f"{month}_clean.csv"
        if not csv_path.exists():
            print(f"skip month={month} missing={csv_path}")
            continue
        print(f"loading {csv_path.name} ...")
        month_hists[month] = load_histogram(csv_path, grid)

    if not month_hists:
        raise FileNotFoundError("No cleaned AIS CSV files were found for the requested months")

    season_prior = normalize01(sum(month_hists.values()))
    out_dir = settings.ais_heatmap_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    month_to_timestamps: dict[str, list[str]] = defaultdict(list)
    for ts in dataset.list_timestamps():
        month_to_timestamps[ts[:7].replace("-", "")].append(ts)

    saved = 0
    for month, hist in month_hists.items():
        heat = build_demo_heatmap(hist, season_prior)
        for ts in month_to_timestamps.get(month, []):
            np.save(out_dir / f"{ts}.npy", heat.astype(np.float32))
            saved += 1

    meta = {
        "tag": args.tag,
        "months": sorted(month_hists.keys()),
        "shape": [h, w],
        "bounds": {
            "lat_min": args.lat_min,
            "lat_max": args.lat_max,
            "lon_min": args.lon_min,
            "lon_max": args.lon_max,
        },
        "saved": saved,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
