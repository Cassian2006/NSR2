from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from app.core import config as config_module
from app.core import dataset as dataset_module
from app.core.latest import _save_annotation_pack
from app.core.source_metadata import (
    build_source_metadata,
    build_source_metadata_report,
    ensure_required_source_metadata,
    validate_source_metadata,
)
from scripts import validate_source_metadata as validate_source_metadata_script


def _prepare_ann_dir(root: Path, ts: str, *, source_meta: dict | None = None) -> None:
    ann = root / "processed" / "annotation_pack" / ts
    ann.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": ts, "channel_names": ["ice_conc"]}
    if source_meta is not None:
        payload["source"] = source_meta
    (ann / "meta.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_ensure_required_source_metadata_backfills_required_fields() -> None:
    out = ensure_required_source_metadata(
        metadata={"source": "copernicus_live"},
        timestamp="2024-07-01_00",
    )
    ok, missing = validate_source_metadata(out)
    assert ok is True
    assert missing == []
    assert out["source"] == "copernicus_live"
    assert out["product_id"] != ""
    assert out["valid_time"] != ""
    assert out["ingested_at"] != ""


def test_build_source_metadata_report_warns_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _prepare_ann_dir(
        data_root,
        "2024-07-01_00",
        source_meta=build_source_metadata(
            source="copernicus_live",
            product_id="cmems_mod_arc_phy_anfc_6km_detided_PT1H-i",
            valid_time="2024-07-01T00:00:00+00:00",
        ),
    )
    _prepare_ann_dir(data_root, "2024-07-01_01", source_meta={"source": "remote_snapshot"})

    monkeypatch.setenv("NSR_DATA_ROOT", str(data_root))
    monkeypatch.setenv("NSR_ALLOW_DEMO_FALLBACK", "0")
    config_module.get_settings.cache_clear()
    try:
        settings = config_module.get_settings()
        report = build_source_metadata_report(settings=settings, sample_limit=10)
    finally:
        config_module.get_settings.cache_clear()

    assert report["summary"]["status"] == "warn"
    assert report["summary"]["sampled_count"] == 2
    assert report["summary"]["ok_count"] == 1
    assert report["summary"]["missing_count"] == 1


def test_latest_save_annotation_pack_always_writes_required_source_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    outputs_root = tmp_path / "outputs"
    monkeypatch.setenv("NSR_DATA_ROOT", str(data_root))
    monkeypatch.setenv("NSR_OUTPUTS_ROOT", str(outputs_root))
    monkeypatch.setenv("NSR_ALLOW_DEMO_FALLBACK", "0")
    config_module.get_settings.cache_clear()
    dataset_module.get_dataset_service.cache_clear()
    try:
        settings = config_module.get_settings()
        _save_annotation_pack(
            settings=settings,
            timestamp="2024-07-01_00",
            x_stack=np.zeros((7, 4, 5), dtype=np.float32),
            blocked_mask=np.zeros((4, 5), dtype=np.uint8),
            channel_names=["ice_conc", "ice_thick", "wave_hs", "wind_u10", "wind_v10", "bathy", "ais_heatmap"],
            source_meta={"source": "remote_snapshot"},
        )
    finally:
        dataset_module.get_dataset_service.cache_clear()
        config_module.get_settings.cache_clear()

    latest_meta_path = data_root / "processed" / "annotation_pack" / "2024-07-01_00" / "latest_meta.json"
    payload = json.loads(latest_meta_path.read_text(encoding="utf-8"))
    ok, missing = validate_source_metadata(payload)
    assert ok is True
    assert missing == []
    assert payload["source"] == "remote_snapshot"


def test_validate_source_metadata_script_generates_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    outputs_root = tmp_path / "outputs"
    _prepare_ann_dir(
        data_root,
        "2024-07-01_00",
        source_meta=build_source_metadata(
            source="copernicus_live",
            product_id="cmems_mod_arc_phy_anfc_6km_detided_PT1H-i",
            valid_time="2024-07-01T00:00:00+00:00",
        ),
    )

    out_dir = tmp_path / "qa"
    monkeypatch.setenv("NSR_DATA_ROOT", str(data_root))
    monkeypatch.setenv("NSR_OUTPUTS_ROOT", str(outputs_root))
    monkeypatch.setenv("NSR_ALLOW_DEMO_FALLBACK", "0")
    config_module.get_settings.cache_clear()
    dataset_module.get_dataset_service.cache_clear()
    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "validate_source_metadata.py",
                "--sample-limit",
                "20",
                "--out-dir",
                str(out_dir),
            ],
        )
        validate_source_metadata_script.main()
    finally:
        dataset_module.get_dataset_service.cache_clear()
        config_module.get_settings.cache_clear()

    reports = sorted(out_dir.glob("source_metadata_*.json"))
    assert reports, "expected source metadata report"
    summary = json.loads(reports[-1].read_text(encoding="utf-8"))["summary"]
    assert summary["status"] == "pass"
