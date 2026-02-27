from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.risk_field import compute_risk_fields, get_risk_layer, get_risk_summary
from app.main import app


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
        env_grids_root=data_root / "interim" / "env_grids",
        dataset_index_path=data_root / "processed" / "dataset" / "index.json",
        timestamps_index_path=data_root / "processed" / "timestamps_index.json",
        grid_spec_path=data_root / "processed" / "grid_spec.json",
        ais_heatmap_root=data_root / "ais_heatmap",
        gallery_root=outputs_root / "gallery",
        pred_root=outputs_root / "pred",
        allow_demo_fallback=False,
        cors_origins="http://localhost:5173",
    )


def test_risk_field_compute_contract_and_cache(tmp_path: Path) -> None:
    settings = _settings_for_tmp(tmp_path)
    ts = "2024-07-01_00"
    h, w = 6, 8
    ann = settings.annotation_pack_root / ts
    ann.mkdir(parents=True, exist_ok=True)

    x_stack = np.zeros((7, h, w), dtype=np.float32)
    x_stack[0] = np.linspace(0, 100, num=h * w, dtype=np.float32).reshape(h, w)  # ice
    x_stack[2] = np.linspace(0, 5, num=h * w, dtype=np.float32).reshape(h, w)  # wave
    x_stack[3] = 3.0
    x_stack[4] = 4.0
    x_stack[5] = np.linspace(0, 1, num=h * w, dtype=np.float32).reshape(h, w)  # ais
    np.save(ann / "x_stack.npy", x_stack)

    blocked = np.zeros((h, w), dtype=np.uint8)
    blocked[0, :] = 1
    np.save(ann / "blocked_mask.npy", blocked)
    (ann / "meta.json").write_text(
        json.dumps(
            {
                "timestamp": ts,
                "channel_names": ["ice_conc", "ice_thick", "wave_hs", "wind_u10", "wind_v10", "ais_heatmap", "bathy"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pred_root = settings.pred_root / "unet_v1"
    pred_root.mkdir(parents=True, exist_ok=True)
    pred = np.zeros((h, w), dtype=np.uint8)
    pred[2:4, 2:5] = 1
    pred[4:, 5:] = 2
    np.save(pred_root / f"{ts}.npy", pred)
    np.save(pred_root / f"{ts}_uncertainty.npy", np.full((h, w), 0.25, dtype=np.float32))

    out = compute_risk_fields(settings=settings, timestamp=ts, force_refresh=True)
    for key in ("risk_mean", "risk_p90", "risk_std"):
        arr = out[key]
        assert arr.shape == (h, w)
        assert arr.dtype == np.float32
        assert float(np.nanmin(arr)) >= 0.0
        assert float(np.nanmax(arr)) <= 1.0
    assert np.allclose(out["risk_mean"][0, :], 1.0)

    out_cached = compute_risk_fields(settings=settings, timestamp=ts, force_refresh=False)
    assert out_cached["meta"].get("cache_hit") is True

    layer = get_risk_layer(settings=settings, timestamp=ts, layer="risk_p90")
    assert layer is not None
    assert layer.shape == (h, w)

    summary = get_risk_summary(settings=settings, timestamp=ts)
    assert summary["shape"] == [h, w]
    assert "risk_mean" in summary["stats"]
    assert summary["meta"]["risk_field_version"] == "risk_v1"


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _pick_timestamp(client: TestClient) -> str:
    resp = client.get("/v1/timestamps")
    assert resp.status_code == 200
    ts = resp.json().get("timestamps", [])
    if not ts:
        pytest.skip("No timestamps available in current dataset")
    return str(ts[0])


def test_risk_summary_api_and_overlay(client: TestClient) -> None:
    ts = _pick_timestamp(client)
    summary_resp = client.get("/v1/risk/summary", params={"timestamp": ts})
    assert summary_resp.status_code == 200
    payload = summary_resp.json()
    assert payload["risk_field_version"] == "risk_v1"
    assert "stats" in payload
    assert "risk_mean" in payload["stats"]

    layers_resp = client.get("/v1/layers", params={"timestamp": ts})
    assert layers_resp.status_code == 200
    layer_ids = {item["id"] for item in layers_resp.json().get("layers", [])}
    assert {"risk_mean", "risk_p90", "risk_std"}.issubset(layer_ids)

    overlay_resp = client.get("/v1/overlay/risk_mean.png", params={"timestamp": ts, "size": "640,360"})
    assert overlay_resp.status_code == 200
    assert overlay_resp.headers["content-type"] == "image/png"
    assert len(overlay_resp.content) > 100

