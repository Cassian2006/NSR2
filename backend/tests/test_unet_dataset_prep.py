from __future__ import annotations

from pathlib import Path

import numpy as np

from app.preprocess.unet_dataset import (
    list_sample_timestamps,
    load_feature_stack,
    make_blocked_mask_from_bathy,
    merge_multiclass_label,
)


def test_list_sample_timestamps_filters_months(tmp_path: Path) -> None:
    samples_root = tmp_path / "samples"
    a = samples_root / "202407" / "2024-07-01_00"
    b = samples_root / "202408" / "2024-08-01_00"
    a.mkdir(parents=True, exist_ok=True)
    b.mkdir(parents=True, exist_ok=True)
    np.save(a / "y_corridor.npy", np.zeros((4, 5), dtype=np.float32))
    np.save(b / "y_corridor.npy", np.zeros((4, 5), dtype=np.float32))

    all_ts = list_sample_timestamps(samples_root)
    july_ts = list_sample_timestamps(samples_root, months={"202407"})
    assert all_ts == ["2024-07-01_00", "2024-08-01_00"]
    assert july_ts == ["2024-07-01_00"]


def test_load_feature_stack_builds_expected_channels(tmp_path: Path) -> None:
    ts = "2024-07-01_00"
    env_dir = tmp_path / "env" / ts
    heat_root = tmp_path / "heat"
    env_dir.mkdir(parents=True, exist_ok=True)
    heat_root.mkdir(parents=True, exist_ok=True)

    x_env = np.ones((5, 4, 6), dtype=np.float32)
    x_bathy = np.full((4, 6), -100.0, dtype=np.float32)
    hm = np.full((4, 6), 0.2, dtype=np.float32)
    np.save(env_dir / "x_env.npy", x_env)
    np.save(env_dir / "x_bathy.npy", x_bathy)
    np.save(heat_root / f"{ts}.npy", hm)
    (env_dir / "meta.json").write_text(
        '{"channels":["ice_conc","ice_thick","wave_hs","wind_u10","wind_v10"]}',
        encoding="utf-8",
    )

    bundle = load_feature_stack(ts, tmp_path / "env", heat_root)
    assert bundle.stack.shape == (7, 4, 6)
    assert bundle.channel_names[-2:] == ["bathy", "ais_heatmap"]
    assert bool(bundle.has_bathy) is True


def test_merge_multiclass_label_blocked_overrides_caution() -> None:
    caution = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    blocked = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    y = merge_multiclass_label(blocked, caution)
    assert y.tolist() == [[2, 1], [2, 0]]


def test_make_blocked_mask_from_bathy_marks_nan_and_threshold() -> None:
    bathy = np.array([[-10.0, 1.0], [np.nan, -2.0]], dtype=np.float32)
    blocked = make_blocked_mask_from_bathy(bathy, blocked_if_bathy_gte=0.0)
    assert blocked.tolist() == [[0, 1], [1, 0]]
