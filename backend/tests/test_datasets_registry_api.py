from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.datasets_registry import build_datasets_registry
from app.main import app


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
        timestamps_index_path=data_root / "processed" / "timestamps_index.json",
        grid_spec_path=data_root / "processed" / "grid_spec.json",
        ais_heatmap_root=data_root / "ais_heatmap",
        gallery_root=outputs_root / "gallery",
        pred_root=outputs_root / "pred",
        cors_origins="http://localhost:5173",
    )


def _prepare_timestamp(
    settings: Settings,
    ts: str,
    *,
    source_meta: dict | None,
    has_ais: bool,
) -> None:
    ann = settings.annotation_pack_root / ts
    ann.mkdir(parents=True, exist_ok=True)
    np.save(ann / "x_stack.npy", np.zeros((7, 4, 5), dtype=np.float32))
    np.save(ann / "blocked_mask.npy", np.zeros((4, 5), dtype=np.uint8))
    payload: dict = {"timestamp": ts}
    if source_meta is not None:
        payload["source"] = source_meta
    (ann / "meta.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    if has_ais:
        hm = settings.ais_heatmap_root / "7d"
        hm.mkdir(parents=True, exist_ok=True)
        np.save(hm / f"{ts}.npy", np.zeros((4, 5), dtype=np.float32))


def test_build_datasets_registry_filters_and_pagination(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    _prepare_timestamp(
        settings,
        "2024-07-01_00",
        source_meta={
            "source": "copernicus_live",
            "product_id": "cmems_a",
            "valid_time": "2024-07-01T00:00:00+00:00",
            "ingested_at": "2026-02-12T00:00:00+00:00",
        },
        has_ais=True,
    )
    _prepare_timestamp(
        settings,
        "2024-07-01_01",
        source_meta={
            "source": "copernicus_live",
            "product_id": "cmems_b",
            "valid_time": "2024-07-01T01:00:00+00:00",
            "ingested_at": "2026-02-12T00:10:00+00:00",
        },
        has_ais=False,
    )
    _prepare_timestamp(
        settings,
        "2024-08-01_00",
        source_meta={"source": "remote_snapshot"},
        has_ais=True,
    )

    all_rows = build_datasets_registry(settings=settings, page=1, page_size=100)
    assert all_rows["summary"]["total_samples"] == 3
    assert all_rows["summary"]["complete_samples"] == 1
    assert "2024-07" in all_rows["summary"]["month_coverage"]
    assert "copernicus_live" in all_rows["summary"]["source_coverage_rate"]

    by_month = build_datasets_registry(settings=settings, month="2024-08", page=1, page_size=50)
    assert by_month["summary"]["total_samples"] == 1
    assert by_month["items"][0]["timestamp"] == "2024-08-01_00"

    by_source = build_datasets_registry(settings=settings, source="copernicus_live", page=1, page_size=50)
    assert by_source["summary"]["total_samples"] == 2
    assert all(item["source"] == "copernicus_live" for item in by_source["items"])

    only_complete = build_datasets_registry(settings=settings, is_complete=True, page=1, page_size=50)
    assert only_complete["summary"]["total_samples"] == 1
    assert only_complete["items"][0]["is_complete"] is True

    paged = build_datasets_registry(settings=settings, page=2, page_size=1)
    assert paged["summary"]["page"] == 2
    assert paged["summary"]["total_pages"] == 3
    assert len(paged["items"]) == 1


def test_datasets_registry_api_shape() -> None:
    client = TestClient(app)
    resp = client.get("/v1/datasets/registry", params={"page": 1, "page_size": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert "summary" in body
    assert "filters" in body
    assert "items" in body
    assert "data_version" in body["summary"]
    assert "month_coverage" in body["summary"]
    assert "source_coverage_rate" in body["summary"]
