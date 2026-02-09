from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    height: int
    width: int
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def parse_timestamp(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def discover_grid_shape(samples_root: Path) -> tuple[int, int]:
    for path in samples_root.rglob("y_corridor.npy"):
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D corridor array: {path}")
        return int(arr.shape[0]), int(arr.shape[1])
    raise FileNotFoundError(f"No y_corridor.npy found under {samples_root}")


def make_timestamps_for_month(month: str, step_hours: int) -> list[datetime]:
    start = datetime.strptime(month + "01", "%Y%m%d")
    if start.month == 12:
        end = datetime(year=start.year + 1, month=1, day=1)
    else:
        end = datetime(year=start.year, month=start.month + 1, day=1)
    out: list[datetime] = []
    t = start
    step = timedelta(hours=step_hours)
    while t < end:
        out.append(t)
        t += step
    return out


def to_cell(lat: float, lon: float, grid: GridSpec) -> tuple[int, int] | None:
    if lat < grid.lat_min or lat > grid.lat_max or lon < grid.lon_min or lon > grid.lon_max:
        return None
    lat_span = grid.lat_max - grid.lat_min
    lon_span = grid.lon_max - grid.lon_min
    if lat_span <= 0 or lon_span <= 0:
        return None
    row = int((grid.lat_max - lat) / lat_span * grid.height)
    col = int((lon - grid.lon_min) / lon_span * grid.width)
    row = min(max(row, 0), grid.height - 1)
    col = min(max(col, 0), grid.width - 1)
    return row, col


def normalize_heatmap(hist: np.ndarray) -> np.ndarray:
    out = np.log1p(hist.astype(np.float32))
    max_val = float(out.max())
    if max_val > 0:
        out /= max_val
    return out


def _save_heatmap(out_dir: Path, ts: datetime, hist: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = normalize_heatmap(hist)
    target = out_dir / f"{ts:%Y-%m-%d_%H}.npy"
    np.save(target, out.astype(np.float32))
    return target


def generate_heatmaps_from_csv(
    csv_path: Path,
    out_dir: Path,
    grid: GridSpec,
    timestamps: Iterable[datetime],
    window_hours: int = 168,
) -> list[Path]:
    centers = sorted(timestamps)
    if not centers:
        return []

    events: list[tuple[datetime, int, int]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            postime = parse_timestamp(row["postime"])
            try:
                lon = float(row["lon"])
                lat = float(row["lat"])
            except (TypeError, ValueError):
                continue
            cell = to_cell(lat, lon, grid)
            if cell is None:
                continue
            r, c = cell
            events.append((postime, r, c))

    events.sort(key=lambda x: x[0])

    hist = np.zeros((grid.height, grid.width), dtype=np.int32)
    q: deque[tuple[datetime, int, int]] = deque()
    center_idx = 0
    saved: list[Path] = []
    window = timedelta(hours=window_hours)

    def flush_until(limit: datetime) -> None:
        nonlocal center_idx
        while center_idx < len(centers) and centers[center_idx] <= limit:
            center = centers[center_idx]
            min_time = center - window
            while q and q[0][0] < min_time:
                _, r0, c0 = q.popleft()
                hist[r0, c0] -= 1
            saved.append(_save_heatmap(out_dir, center, hist))
            center_idx += 1

    for postime, r, c in events:
        flush_until(postime)
        q.append((postime, r, c))
        hist[r, c] += 1

    flush_until(datetime.max)
    return saved
