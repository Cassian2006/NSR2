from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.ais_display import build_ais_display_grid


def test_build_ais_display_grid_expands_sparse_local_signal(tmp_path: Path) -> None:
    ais_root = tmp_path / "ais_heatmap" / "7d"
    ais_root.mkdir(parents=True, exist_ok=True)

    hist = np.zeros((8, 8), dtype=np.float32)
    hist[2, 2] = 1.0
    hist[5, 5] = 0.8
    np.save(ais_root / "2024-07-01_00.npy", hist)
    np.save(ais_root / "2024-07-02_00.npy", hist)

    local = np.zeros((8, 8), dtype=np.float32)
    local[2, 2] = 1.0

    display = build_ais_display_grid(ais_root=ais_root.parent, local_grid=local, shape=local.shape)
    assert display is not None
    assert display.shape == local.shape
    assert float(np.mean(display > 1e-6)) > float(np.mean(local > 1e-6))
    assert float(display[2, 2]) >= 0.99


def test_build_ais_display_grid_can_fall_back_to_prior_only(tmp_path: Path) -> None:
    ais_root = tmp_path / "ais_heatmap" / "7d"
    ais_root.mkdir(parents=True, exist_ok=True)

    hist = np.zeros((6, 6), dtype=np.float32)
    hist[3, 3] = 1.0
    np.save(ais_root / "2024-08-01_00.npy", hist)

    display = build_ais_display_grid(ais_root=ais_root.parent, shape=(6, 6))
    assert display is not None
    assert float(np.max(display)) > 0.0
