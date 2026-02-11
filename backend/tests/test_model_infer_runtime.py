from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from app.core.config import Settings
from app.model.infer import run_unet_inference
from app.model.tiny_unet import TinyUNet


def _build_test_settings(tmp_path: Path, summary_path: Path) -> Settings:
    project_root = tmp_path
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    annotation_pack_root = data_root / "processed" / "annotation_pack"
    pred_root = outputs_root / "pred"
    return Settings(
        project_root=project_root,
        data_root=data_root,
        outputs_root=outputs_root,
        processed_samples_root=data_root / "processed" / "samples",
        annotation_pack_root=annotation_pack_root,
        dataset_index_path=data_root / "processed" / "dataset" / "index.json",
        ais_heatmap_root=data_root / "ais_heatmap",
        gallery_root=outputs_root / "gallery",
        pred_root=pred_root,
        unet_default_summary=summary_path,
        cors_origins="http://localhost:5173",
    )


def test_run_unet_inference_persists_and_hits_cache(tmp_path: Path) -> None:
    ts = "2024-07-01_00"
    annotation_dir = tmp_path / "data" / "processed" / "annotation_pack" / ts
    annotation_dir.mkdir(parents=True, exist_ok=True)
    x = np.random.default_rng(0).normal(0.0, 1.0, size=(7, 24, 24)).astype(np.float32)
    np.save(annotation_dir / "x_stack.npy", x)

    model = TinyUNet(in_channels=7, n_classes=3, base=24)
    ckpt_path = tmp_path / "outputs" / "train_runs" / "fake_run" / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": 1}, ckpt_path)

    summary_path = ckpt_path.parent / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "in_channels": 7,
                "norm_mean": [0.0] * 7,
                "norm_std": [1.0] * 7,
                "best_ckpt": str(ckpt_path),
            }
        ),
        encoding="utf-8",
    )

    settings = _build_test_settings(tmp_path=tmp_path, summary_path=summary_path)
    out_file = settings.pred_root / "unet_v1" / f"{ts}.npy"

    first = run_unet_inference(
        settings=settings,
        timestamp=ts,
        model_version="unet_v1",
        output_path=out_file,
    )
    assert out_file.exists()
    unc_file = settings.pred_root / "unet_v1" / f"{ts}_uncertainty.npy"
    assert unc_file.exists()
    assert first["cache_hit"] is False
    assert first["shape"] == [24, 24]
    assert first["uncertainty_file"].endswith(f"{ts}_uncertainty.npy")
    assert isinstance(first["uncertainty_mean"], float)
    assert isinstance(first["uncertainty_p90"], float)

    second = run_unet_inference(
        settings=settings,
        timestamp=ts,
        model_version="unet_v1",
        output_path=out_file,
    )
    assert second["cache_hit"] is True
    assert second["class_hist"]["safe"] + second["class_hist"]["caution"] + second["class_hist"]["blocked"] == 24 * 24
    assert second["uncertainty_file"].endswith(f"{ts}_uncertainty.npy")


def test_run_unet_inference_uses_heuristic_fallback_when_torch_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ts = "2024-07-01_06"
    annotation_dir = tmp_path / "data" / "processed" / "annotation_pack" / ts
    annotation_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, size=(7, 24, 24)).astype(np.float32)
    np.save(annotation_dir / "x_stack.npy", x)
    blocked = np.zeros((24, 24), dtype=np.uint8)
    blocked[:3, :] = 1
    np.save(annotation_dir / "blocked_mask.npy", blocked)
    (annotation_dir / "meta.json").write_text(
        json.dumps(
            {
                "channel_names": [
                    "ice_conc",
                    "ice_thick",
                    "wave_hs",
                    "wind_u10",
                    "wind_v10",
                    "bathy",
                    "ais_heatmap",
                ]
            }
        ),
        encoding="utf-8",
    )

    summary_path = tmp_path / "outputs" / "train_runs" / "fake_run" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("{}", encoding="utf-8")
    settings = _build_test_settings(tmp_path=tmp_path, summary_path=summary_path)
    out_file = settings.pred_root / "unet_v1" / f"{ts}.npy"

    import app.model.infer as infer_module

    monkeypatch.setattr(infer_module, "torch", None, raising=False)
    monkeypatch.setattr(infer_module, "_torch_import_error", ModuleNotFoundError("torch unavailable"), raising=False)

    stats = run_unet_inference(
        settings=settings,
        timestamp=ts,
        model_version="unet_v1",
        output_path=out_file,
    )
    assert stats["fallback_mode"] == "heuristic_no_torch"
    assert out_file.exists()
    unc_file = settings.pred_root / "unet_v1" / f"{ts}_uncertainty.npy"
    assert unc_file.exists()
