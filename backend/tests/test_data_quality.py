from __future__ import annotations

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
        ais_heatmap_root=data_root / "ais_heatmap",
        gallery_root=outputs_root / "gallery",
        pred_root=outputs_root / "pred",
        unet_default_summary=outputs_root / "train_runs" / "fake" / "summary.json",
        cors_origins="http://localhost:5173",
    )


def test_build_data_quality_report_basic(tmp_path: Path) -> None:
    settings = _settings_for_tmp(tmp_path)
    ts = "2024-07-01_00"
    ann = settings.annotation_pack_root / ts
    ann.mkdir(parents=True, exist_ok=True)
    np.save(ann / "x_stack.npy", np.zeros((7, 8, 8), dtype=np.float32))
    np.save(ann / "blocked_mask.npy", np.zeros((8, 8), dtype=np.uint8))
    settings.ais_heatmap_root.mkdir(parents=True, exist_ok=True)
    (settings.ais_heatmap_root / "demo").mkdir(parents=True, exist_ok=True)
    np.save(settings.ais_heatmap_root / "demo" / f"{ts}.npy", np.zeros((8, 8), dtype=np.float32))

    report = build_data_quality_report(settings=settings, sample_limit=16)
    assert report["summary"]["timestamp_count"] == 1
    assert isinstance(report.get("checks"), list)
    assert len(report["checks"]) >= 3
    statuses = {c["status"] for c in report["checks"]}
    assert statuses.issubset({"pass", "warn", "fail"})
