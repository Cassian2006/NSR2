from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.core.annotation_workspace import AnnotationWorkspaceService
from app.core.config import Settings
from app.core.geo import load_grid_geo


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


def _seed_annotation_sample(settings: Settings, ts: str, h: int = 24, w: int = 36) -> Path:
    ann = settings.annotation_pack_root / ts
    ann.mkdir(parents=True, exist_ok=True)
    x = np.zeros((7, h, w), dtype=np.float32)
    blocked = np.zeros((h, w), dtype=np.uint8)
    blocked[:2, :] = 1
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
    return ann


def _rect_points_from_cells(settings: Settings, ts: str, shape: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> list[dict[str, float]]:
    geo = load_grid_geo(settings, timestamp=ts, shape=shape)
    lat0, lon0 = geo.rc_to_latlon(r0, c0)
    lat1, lon1 = geo.rc_to_latlon(r1, c0)
    lat2, lon2 = geo.rc_to_latlon(r1, c1)
    lat3, lon3 = geo.rc_to_latlon(r0, c1)
    return [
        {"lat": float(lat0), "lon": float(lon0)},
        {"lat": float(lat1), "lon": float(lon1)},
        {"lat": float(lat2), "lon": float(lon2)},
        {"lat": float(lat3), "lon": float(lon3)},
    ]


def test_annotation_workspace_save_and_get_patch(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    ts = "2024-07-01_00"
    ann = _seed_annotation_sample(settings, ts)
    service = AnnotationWorkspaceService(settings=settings)

    initial = service.get_patch(ts)
    assert initial["stats"]["caution_pixels"] == 0
    assert initial["stats"]["operations_count"] == 0

    shape = tuple(np.load(ann / "blocked_mask.npy").shape)
    points = _rect_points_from_cells(settings, ts, shape, r0=6, c0=8, r1=13, c1=18)
    saved = service.save_patch(
        timestamp_raw=ts,
        operations_raw=[{"id": "op-1", "mode": "add", "points": points}],
        note="test add",
        author="pytest",
    )

    assert saved["stats"]["caution_pixels"] > 0
    assert saved["stats"]["operations_count"] == 1
    assert Path(saved["patch_file"]).exists()
    assert Path(saved["caution_file"]).exists()
    assert Path(saved["y_class_file"]).exists()

    caution = np.load(saved["caution_file"])
    blocked = np.load(ann / "blocked_mask.npy")
    # blocked cells must remain non-caution
    assert int((caution[blocked > 0] > 0).sum()) == 0

    y = np.load(saved["y_class_file"])
    assert y.shape == caution.shape
    assert int((y == 1).sum()) == int((caution > 0).sum())
    assert int((y == 2).sum()) == int((blocked > 0).sum())

    loaded = service.get_patch(ts)
    assert loaded["stats"]["operations_count"] == 1
    assert loaded["operations"][0]["mode"] == "add"


def test_annotation_workspace_erase_operation(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    ts = "2024-07-02_00"
    ann = _seed_annotation_sample(settings, ts)
    service = AnnotationWorkspaceService(settings=settings)
    shape = tuple(np.load(ann / "blocked_mask.npy").shape)
    points = _rect_points_from_cells(settings, ts, shape, r0=6, c0=8, r1=13, c1=18)

    first = service.save_patch(
        timestamp_raw=ts,
        operations_raw=[{"id": "op-add", "mode": "add", "points": points}],
        note="add",
    )
    assert first["stats"]["caution_pixels"] > 0

    second = service.save_patch(
        timestamp_raw=ts,
        operations_raw=[
            {"id": "op-add", "mode": "add", "points": points},
            {"id": "op-erase", "mode": "erase", "points": points},
        ],
        note="erase",
    )
    assert second["stats"]["caution_pixels"] == 0


def test_annotation_workspace_stroke_operation(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    ts = "2024-07-03_00"
    ann = _seed_annotation_sample(settings, ts)
    service = AnnotationWorkspaceService(settings=settings)
    shape = tuple(np.load(ann / "blocked_mask.npy").shape)
    stroke = _rect_points_from_cells(settings, ts, shape, r0=8, c0=4, r1=8, c1=24)[:2]
    saved = service.save_patch(
        timestamp_raw=ts,
        operations_raw=[
            {
                "id": "op-stroke",
                "mode": "add",
                "shape": "stroke",
                "radius_cells": 3,
                "points": stroke,
            }
        ],
        note="stroke",
    )
    assert saved["stats"]["caution_pixels"] > 0
    caution = np.load(saved["caution_file"])
    assert int((caution > 0).sum()) >= 20
