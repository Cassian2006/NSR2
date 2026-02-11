from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.core.dataset import DatasetService


def _build_settings(tmp_path: Path) -> Settings:
    project_root = tmp_path
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    return Settings(
        project_root=project_root,
        data_root=data_root,
        outputs_root=outputs_root,
        processed_samples_root=data_root / "processed" / "samples",
        annotation_pack_root=data_root / "processed" / "annotation_pack",
        env_grids_root=data_root / "interim" / "env_grids",
        dataset_index_path=data_root / "processed" / "dataset" / "index.json",
        ais_heatmap_root=data_root / "ais_heatmap",
        gallery_root=outputs_root / "gallery",
        pred_root=outputs_root / "pred",
        cors_origins="http://localhost:5173",
    )


def test_list_layers_uses_annotation_pack_channels_for_availability(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    ts = "2024-07-01_00"
    ann = settings.annotation_pack_root / ts
    ann.mkdir(parents=True, exist_ok=True)
    x = np.zeros((7, 8, 8), dtype=np.float32)
    blocked = np.zeros((8, 8), dtype=np.uint8)
    np.save(ann / "x_stack.npy", x)
    np.save(ann / "blocked_mask.npy", blocked)
    (ann / "meta.json").write_text(
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

    svc = DatasetService()
    svc.settings = settings
    layers = {item["id"]: item for item in svc.list_layers(ts)}

    assert layers["unet_pred"]["available"] is True
    assert layers["unet_uncertainty"]["available"] is True
    assert layers["ice"]["available"] is True
    assert layers["wave"]["available"] is True
    assert layers["wind"]["available"] is True
