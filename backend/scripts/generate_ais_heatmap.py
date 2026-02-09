from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.preprocess.ais_heatmap import (
    GridSpec,
    discover_grid_shape,
    generate_heatmaps_from_csv,
    make_timestamps_for_month,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate AIS heatmaps from cleaned AIS CSV")
    p.add_argument("--month", required=True, help="Month in YYYYMM format, e.g. 202407")
    p.add_argument("--window-hours", type=int, default=168, help="Rolling window size in hours")
    p.add_argument("--step-hours", type=int, default=6, help="Output step size in hours")
    p.add_argument("--tag", default="7d", help="Output subfolder under data/ais_heatmap/")
    p.add_argument("--height", type=int, default=0, help="Grid height; 0 means auto-discover")
    p.add_argument("--width", type=int, default=0, help="Grid width; 0 means auto-discover")
    p.add_argument("--lat-min", type=float, default=60.0)
    p.add_argument("--lat-max", type=float, default=86.0)
    p.add_argument("--lon-min", type=float, default=-180.0)
    p.add_argument("--lon-max", type=float, default=180.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    csv_path = settings.data_root / "processed" / "ais_cleaned" / f"{args.month}_clean.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned AIS file not found: {csv_path}")

    if args.height > 0 and args.width > 0:
        h, w = args.height, args.width
    else:
        h, w = discover_grid_shape(settings.processed_samples_root)

    grid = GridSpec(
        height=h,
        width=w,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
    )
    timestamps = make_timestamps_for_month(args.month, args.step_hours)
    out_dir = settings.ais_heatmap_root / args.tag
    saved = generate_heatmaps_from_csv(
        csv_path=Path(csv_path),
        out_dir=out_dir,
        grid=grid,
        timestamps=timestamps,
        window_hours=args.window_hours,
    )
    print(f"saved={len(saved)} out_dir={out_dir}")
    if saved:
        print(f"first={saved[0].name} last={saved[-1].name}")


if __name__ == "__main__":
    main()
