from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.core.data_quality import build_data_quality_report


def _settings_for_tmp(tmp_path: Path) -> Settings:
    project_root = tmp_path
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    return Settings(
        project_root=project_root,
        data_root=data_root,
        outputs_root=outputs_root,
        processed_samples_root=data_root / "processed" / "samples",
        annotation_pack_root=data_root / "processed" / "annotation_pack",
        dataset_index_path=data_root / "processed" / "dataset" / "index.json",
        timestamps_index_path=data_root / "processed" / "timestamps_index.json",
        grid_spec_path=data_root / "processed" / "grid_spec.json",
        ais_heatmap_root=data_root / "ais_heatmap",
        gallery_root=outputs_root / "gallery",
        pred_root=outputs_root / "pred",
        unet_default_summary=outputs_root / "train_runs" / "fake" / "summary.json",
        allow_demo_fallback=False,
        cors_origins="http://localhost:5173",
    )


def _write_pack(settings: Settings, ts: str, *, h: int = 8, w: int = 8) -> None:
    ann = settings.annotation_pack_root / ts
    ann.mkdir(parents=True, exist_ok=True)
    np.save(ann / "x_stack.npy", np.zeros((7, h, w), dtype=np.float32))
    np.save(ann / "blocked_mask.npy", np.zeros((h, w), dtype=np.uint8))


def test_quality_gate_fail_on_critical_shape_mismatch(tmp_path: Path) -> None:
    settings = _settings_for_tmp(tmp_path)
    _write_pack(settings, "2024-07-01_00", h=8, w=8)
    _write_pack(settings, "2024-07-01_01", h=10, w=8)
    report = build_data_quality_report(settings=settings, sample_limit=16)
    assert report["gate"]["status"] == "FAIL"
    assert report["gate"]["block_release"] is True
    assert "shape_consistency" in report["gate"]["blockers"]

    shape_check = next(c for c in report["checks"] if c["name"] == "shape_consistency")
    assert shape_check["severity"] == "critical"
    assert shape_check["status"] == "fail"
    assert shape_check["action"] == "block_release"


def test_quality_gate_warn_on_degradable_aux_missing(tmp_path: Path) -> None:
    settings = _settings_for_tmp(tmp_path)
    _write_pack(settings, "2024-07-01_00")
    _write_pack(settings, "2024-07-01_01")
    report = build_data_quality_report(settings=settings, sample_limit=16)
    assert report["gate"]["status"] == "WARN"
    assert report["gate"]["block_release"] is False
    assert report["gate"]["blockers"] == []

    aux = next(c for c in report["checks"] if c["name"] == "auxiliary_layers_coverage")
    assert aux["severity"] == "degradable"
    assert aux["raw_status"] == "fail"
    assert aux["status"] == "warn"


def test_quality_gate_pass_when_all_core_checks_pass(tmp_path: Path) -> None:
    settings = _settings_for_tmp(tmp_path)
    _write_pack(settings, "2024-07-01_00")
    _write_pack(settings, "2024-07-01_01")

    hm_root = settings.ais_heatmap_root / "7d"
    hm_root.mkdir(parents=True, exist_ok=True)
    np.save(hm_root / "2024-07-01_00.npy", np.zeros((8, 8), dtype=np.float32))
    np.save(hm_root / "2024-07-01_01.npy", np.zeros((8, 8), dtype=np.float32))
    pred_root = settings.pred_root / "unet_v1"
    pred_root.mkdir(parents=True, exist_ok=True)
    np.save(pred_root / "2024-07-01_00.npy", np.zeros((8, 8), dtype=np.uint8))
    np.save(pred_root / "2024-07-01_01.npy", np.zeros((8, 8), dtype=np.uint8))
    np.save(pred_root / "2024-07-01_00_uncertainty.npy", np.zeros((8, 8), dtype=np.float32))
    np.save(pred_root / "2024-07-01_01_uncertainty.npy", np.zeros((8, 8), dtype=np.float32))

    report = build_data_quality_report(settings=settings, sample_limit=16)
    assert report["gate"]["status"] == "PASS"
    assert report["gate"]["block_release"] is False
    assert report["summary"]["status"] == "pass"


def test_report_data_quality_script_enforce_gate_fail_exit_code(tmp_path: Path) -> None:
    backend_root = Path(__file__).resolve().parents[1]
    script = backend_root / "scripts" / "report_data_quality.py"
    data_root = tmp_path / "data"
    outputs_root = tmp_path / "outputs"
    env = os.environ.copy()
    env["NSR_DATA_ROOT"] = str(data_root)
    env["NSR_OUTPUTS_ROOT"] = str(outputs_root)
    env["NSR_ALLOW_DEMO_FALLBACK"] = "false"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--sample-limit",
            "16",
            "--out-dir",
            str(outputs_root / "qa"),
            "--enforce-gate",
        ],
        cwd=str(backend_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "gate_status=FAIL" in proc.stdout
