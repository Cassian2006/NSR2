from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core import config as config_module


def test_data_root_override_updates_derived_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NSR_PROJECT_ROOT", str(tmp_path / "isolated_project_root"))
    custom_data_root = tmp_path / "custom_data"
    monkeypatch.setenv("NSR_DATA_ROOT", str(custom_data_root))
    monkeypatch.delenv("NSR_ANNOTATION_PACK_ROOT", raising=False)
    monkeypatch.delenv("NSR_AIS_HEATMAP_ROOT", raising=False)

    config_module.get_settings.cache_clear()
    try:
        settings = config_module.get_settings()
        assert settings.data_root == custom_data_root
        assert settings.annotation_pack_root == custom_data_root / "processed" / "annotation_pack"
        assert settings.ais_heatmap_root == custom_data_root / "ais_heatmap"
    finally:
        config_module.get_settings.cache_clear()


def test_fallback_to_demo_data_when_configured_roots_have_no_samples(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    demo_ts_dir = project_root / "backend" / "demo_data" / "processed" / "annotation_pack" / "2024-07-15_00"
    demo_heatmap_dir = project_root / "backend" / "demo_data" / "ais_heatmap" / "7d"
    demo_ts_dir.mkdir(parents=True, exist_ok=True)
    demo_heatmap_dir.mkdir(parents=True, exist_ok=True)
    np.save(demo_ts_dir / "x_stack.npy", np.zeros((7, 8, 9), dtype=np.float32))
    np.save(demo_ts_dir / "blocked_mask.npy", np.zeros((8, 9), dtype=np.uint8))
    np.save(demo_heatmap_dir / "2024-07-15_00.npy", np.zeros((8, 9), dtype=np.float32))

    bad_data_root = project_root / "missing_data_root"
    monkeypatch.setenv("NSR_PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("NSR_DATA_ROOT", str(bad_data_root))
    monkeypatch.delenv("NSR_ANNOTATION_PACK_ROOT", raising=False)
    monkeypatch.delenv("NSR_AIS_HEATMAP_ROOT", raising=False)

    config_module.get_settings.cache_clear()
    try:
        settings = config_module.get_settings()
        assert settings.data_root == project_root / "backend" / "demo_data"
        assert settings.annotation_pack_root == project_root / "backend" / "demo_data" / "processed" / "annotation_pack"
        assert settings.ais_heatmap_root == project_root / "backend" / "demo_data" / "ais_heatmap"
    finally:
        config_module.get_settings.cache_clear()
