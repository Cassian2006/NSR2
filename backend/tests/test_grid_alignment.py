from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from app.core import config as config_module
from app.core.grid_alignment import build_grid_alignment_report, get_timestamp_alignment
from scripts import check_grid_alignment as check_grid_alignment_script


def _prepare_timestamp(
    data_root: Path,
    outputs_root: Path,
    ts: str,
    *,
    hw: tuple[int, int] = (6, 8),
    ais_hw: tuple[int, int] | None = None,
) -> None:
    h, w = hw
    ann = data_root / "processed" / "annotation_pack" / ts
    ann.mkdir(parents=True, exist_ok=True)
    np.save(ann / "x_stack.npy", np.zeros((7, h, w), dtype=np.float32))
    np.save(ann / "blocked_mask.npy", np.zeros((h, w), dtype=np.uint8))

    ais = data_root / "ais_heatmap" / "7d"
    ais.mkdir(parents=True, exist_ok=True)
    ah, aw = ais_hw if ais_hw is not None else (h, w)
    np.save(ais / f"{ts}.npy", np.zeros((ah, aw), dtype=np.float32))

    pred = outputs_root / "pred" / "unet_v1"
    pred.mkdir(parents=True, exist_ok=True)
    np.save(pred / f"{ts}.npy", np.zeros((h, w), dtype=np.uint8))
    np.save(pred / f"{ts}_uncertainty.npy", np.zeros((h, w), dtype=np.float32))


def test_grid_alignment_report_warn_on_shape_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    outputs_root = tmp_path / "outputs"
    _prepare_timestamp(data_root, outputs_root, "2024-07-01_00", hw=(6, 8))
    _prepare_timestamp(data_root, outputs_root, "2024-07-01_01", hw=(6, 8), ais_hw=(7, 8))

    monkeypatch.setenv("NSR_DATA_ROOT", str(data_root))
    monkeypatch.setenv("NSR_OUTPUTS_ROOT", str(outputs_root))
    monkeypatch.setenv("NSR_ALLOW_DEMO_FALLBACK", "0")
    config_module.get_settings.cache_clear()
    try:
        settings = config_module.get_settings()
        report = build_grid_alignment_report(settings=settings, sample_limit=10)
        assert report["summary"]["status"] == "warn"
        assert report["summary"]["mismatch_count"] == 1
        mismatch = report["mismatches"][0]
        assert mismatch["timestamp"] == "2024-07-01_01"
        detail = get_timestamp_alignment(settings, "2024-07-01_01")
        assert detail["ok"] is False
        assert "ais_heatmap" in detail["mismatch_layers"]
    finally:
        config_module.get_settings.cache_clear()


def test_check_grid_alignment_script_writes_grid_spec(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    outputs_root = tmp_path / "outputs"
    _prepare_timestamp(data_root, outputs_root, "2024-07-01_00", hw=(10, 12))
    _prepare_timestamp(data_root, outputs_root, "2024-07-01_01", hw=(10, 12))

    grid_spec_out = tmp_path / "grid_spec.json"
    report_out = tmp_path / "qa"
    monkeypatch.setenv("NSR_DATA_ROOT", str(data_root))
    monkeypatch.setenv("NSR_OUTPUTS_ROOT", str(outputs_root))
    monkeypatch.setenv("NSR_ALLOW_DEMO_FALLBACK", "0")
    config_module.get_settings.cache_clear()
    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "check_grid_alignment.py",
                "--sample-limit",
                "10",
                "--grid-spec-out",
                str(grid_spec_out),
                "--report-out-dir",
                str(report_out),
            ],
        )
        check_grid_alignment_script.main()
    finally:
        config_module.get_settings.cache_clear()

    spec = json.loads(grid_spec_out.read_text(encoding="utf-8"))
    assert spec["crs"]["compute"] == "EPSG:3413"
    assert spec["crs"]["display"] == "EPSG:4326"
    assert spec["grid"]["shape"] == [10, 12]
    reports = list(report_out.glob("grid_alignment_*.json"))
    assert reports, "expected grid alignment json report"
