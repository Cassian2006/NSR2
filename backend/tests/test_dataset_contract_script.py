from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from app.core import config as config_module
from scripts import check_dataset_contract as contract_script


def _latest_report(out_dir: Path) -> dict:
    reports = sorted(out_dir.glob("dataset_contract_*.json"))
    assert reports, "expected dataset_contract report json"
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def _prepare_sample(data_root: Path, outputs_root: Path, ts: str, *, shape: tuple[int, int] = (8, 9)) -> None:
    ann = data_root / "processed" / "annotation_pack" / ts
    ann.mkdir(parents=True, exist_ok=True)
    np.save(ann / "x_stack.npy", np.zeros((7, shape[0], shape[1]), dtype=np.float32))
    np.save(ann / "blocked_mask.npy", np.zeros(shape, dtype=np.uint8))
    (ann / "meta.json").write_text(json.dumps({"channel_names": ["a"] * 7}), encoding="utf-8")

    hm = data_root / "ais_heatmap" / "7d"
    hm.mkdir(parents=True, exist_ok=True)
    np.save(hm / f"{ts}.npy", np.zeros(shape, dtype=np.float32))

    pred = outputs_root / "pred" / "unet_v1"
    pred.mkdir(parents=True, exist_ok=True)
    np.save(pred / f"{ts}.npy", np.zeros(shape, dtype=np.uint8))
    np.save(pred / f"{ts}_uncertainty.npy", np.full(shape, 0.2, dtype=np.float32))


def test_check_dataset_contract_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    outputs_root = tmp_path / "outputs"
    _prepare_sample(data_root, outputs_root, "2024-07-01_00")
    _prepare_sample(data_root, outputs_root, "2024-07-01_06")
    _prepare_sample(data_root, outputs_root, "2024-07-01_12")

    out_dir = tmp_path / "qa"
    monkeypatch.setenv("NSR_DATA_ROOT", str(data_root))
    monkeypatch.setenv("NSR_OUTPUTS_ROOT", str(outputs_root))
    monkeypatch.setenv("NSR_ALLOW_DEMO_FALLBACK", "0")
    config_module.get_settings.cache_clear()
    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "check_dataset_contract.py",
                "--sample-limit",
                "2",
                "--out-dir",
                str(out_dir),
            ],
        )
        contract_script.main()
    finally:
        config_module.get_settings.cache_clear()

    report = _latest_report(out_dir)
    assert report["summary"]["status"] == "pass"
    assert report["summary"]["sampled_count"] == 2


def test_check_dataset_contract_fail_on_shape_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    outputs_root = tmp_path / "outputs"
    ts = "2024-07-02_00"
    _prepare_sample(data_root, outputs_root, ts, shape=(10, 12))

    # Force contract violation: blocked_mask shape mismatch with x_stack HxW.
    ann = data_root / "processed" / "annotation_pack" / ts
    np.save(ann / "blocked_mask.npy", np.zeros((9, 12), dtype=np.uint8))

    out_dir = tmp_path / "qa"
    monkeypatch.setenv("NSR_DATA_ROOT", str(data_root))
    monkeypatch.setenv("NSR_OUTPUTS_ROOT", str(outputs_root))
    monkeypatch.setenv("NSR_ALLOW_DEMO_FALLBACK", "0")
    config_module.get_settings.cache_clear()
    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "check_dataset_contract.py",
                "--sample-limit",
                "5",
                "--out-dir",
                str(out_dir),
            ],
        )
        with pytest.raises(SystemExit) as exc:
            contract_script.main()
        assert exc.value.code == 2
    finally:
        config_module.get_settings.cache_clear()

    report = _latest_report(out_dir)
    assert report["summary"]["status"] == "fail"
    checks = {c["name"]: c for c in report["checks"]}
    assert checks["sample_shape_dtype_consistency"]["status"] == "fail"
