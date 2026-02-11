from __future__ import annotations

from pathlib import Path

from app.core import config as config_module


def test_data_root_override_updates_derived_paths(monkeypatch, tmp_path: Path) -> None:
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
