from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from app.preprocess.ais_heatmap import (
    GridSpec,
    generate_heatmaps_from_csv,
    make_timestamps_for_month,
    normalize_heatmap,
    to_cell,
)


def test_to_cell_bounds() -> None:
    grid = GridSpec(height=10, width=20, lat_min=60.0, lat_max=80.0, lon_min=0.0, lon_max=40.0)
    assert to_cell(70.0, 20.0, grid) is not None
    assert to_cell(59.9, 20.0, grid) is None
    assert to_cell(70.0, 40.1, grid) is None


def test_normalize_heatmap() -> None:
    arr = np.array([[0, 1], [2, 3]], dtype=np.int32)
    out = normalize_heatmap(arr)
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) == 1.0


def test_make_timestamps_for_month() -> None:
    july = make_timestamps_for_month("202407", step_hours=6)
    assert july[0] == datetime(2024, 7, 1, 0, 0, 0)
    assert july[-1] == datetime(2024, 7, 31, 18, 0, 0)
    assert len(july) == 124


def test_generate_heatmaps_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "ais.csv"
    csv_path.write_text(
        "\n".join(
            [
                "mmsi,postime,lon,lat,status,eta,dest,draught,cog,hdg,sog,rot",
                "1,2024-07-01 00:00:00,10.0,70.0,0,,,,,,",
                "1,2024-07-01 06:00:00,11.0,70.2,0,,,,,,",
                "2,2024-07-01 12:00:00,12.0,70.4,0,,,,,,",
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "hm"
    grid = GridSpec(height=8, width=8, lat_min=60.0, lat_max=80.0, lon_min=0.0, lon_max=20.0)
    timestamps = [datetime(2024, 7, 1, 6, 0, 0), datetime(2024, 7, 1, 12, 0, 0)]
    saved = generate_heatmaps_from_csv(csv_path, out_dir, grid, timestamps, window_hours=12)
    assert len(saved) == 2
    arr0 = np.load(saved[0])
    arr1 = np.load(saved[1])
    assert arr0.shape == (8, 8)
    assert arr1.shape == (8, 8)
    assert float(arr0.max()) <= 1.0
    assert float(arr1.max()) <= 1.0
    assert float(arr1.sum()) > 0.0
