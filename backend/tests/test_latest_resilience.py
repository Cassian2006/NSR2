from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.core.config import Settings
from app.core.latest import LatestDataError, _with_retries, resolve_latest_timestamp
from app.core.latest_source_health import configure_source_health


def _build_settings(tmp_path: Path) -> Settings:
    project_root = tmp_path
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    annotation_pack_root = data_root / "processed" / "annotation_pack"
    settings = Settings(
        project_root=project_root,
        data_root=data_root,
        outputs_root=outputs_root,
        processed_samples_root=data_root / "processed" / "samples",
        annotation_pack_root=annotation_pack_root,
        dataset_index_path=data_root / "processed" / "dataset" / "index.json",
        ais_heatmap_root=data_root / "ais_heatmap",
        gallery_root=outputs_root / "gallery",
        pred_root=outputs_root / "pred",
        latest_root=outputs_root / "latest",
        latest_progress_store_path=outputs_root / "latest" / "progress_state.json",
        latest_source_health_path=outputs_root / "latest" / "source_health.json",
        latest_snapshot_url_template="",
        copernicus_username="",
        copernicus_password="",
        cors_origins="http://localhost:5173",
        allow_demo_fallback=False,
    )
    return settings


def _write_pack(root: Path, ts: str, channels: int = 7, h: int = 12, w: int = 16) -> None:
    pack = root / ts
    pack.mkdir(parents=True, exist_ok=True)
    np.save(pack / "x_stack.npy", np.zeros((channels, h, w), dtype=np.float32))
    np.save(pack / "blocked_mask.npy", np.zeros((h, w), dtype=np.uint8))
    (pack / "meta.json").write_text(
        json.dumps({"channel_names": ["ice_conc", "ice_thick", "wave_hs", "wind_u10", "wind_v10", "bathy", "ais_heatmap"]}),
        encoding="utf-8",
    )


def test_force_refresh_falls_back_to_stale_local_when_sources_unavailable(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    configure_source_health(
        store_path=settings.latest_source_health_path,
        failure_threshold=3,
        cooldown_sec=60,
    )
    _write_pack(settings.annotation_pack_root, "2024-07-01_12")
    _write_pack(settings.annotation_pack_root, "2024-07-01_00")

    resolved = resolve_latest_timestamp(
        settings=settings,
        date="2024-07-01",
        hour=12,
        force_refresh=True,
    )

    assert resolved.timestamp == "2024-07-01_12"
    assert resolved.source == "stale_local_existing"
    assert "snapshot_error=" in resolved.note


def test_nearest_fallback_uses_passed_settings_local_timestamps(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    configure_source_health(
        store_path=settings.latest_source_health_path,
        failure_threshold=3,
        cooldown_sec=60,
    )
    _write_pack(settings.annotation_pack_root, "2024-07-10_12")
    _write_pack(settings.annotation_pack_root, "2024-07-20_12")

    resolved = resolve_latest_timestamp(
        settings=settings,
        date="2024-07-01",
        hour=12,
        force_refresh=True,
    )
    assert resolved.source == "nearest_local_fallback"
    assert resolved.timestamp == "2024-07-10_12"


def test_with_retries_stops_on_non_retryable_error() -> None:
    calls = {"count": 0}

    def _run() -> None:
        calls["count"] += 1
        raise LatestDataError("Copernicus auth incomplete: unauthorized")

    with pytest.raises(LatestDataError):
        _with_retries(
            run=_run,
            retries=4,
            backoff_sec=0.01,
            label="non-retryable",
        )
    assert calls["count"] == 1

