from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.dataset import normalize_timestamp
from app.core.geo import load_grid_geo
from app.main import app


TEST_START = {"lat": 70.5, "lon": 30.0}
TEST_GOAL = {"lat": 72.0, "lon": 150.0}


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _pick_timestamp(client: TestClient) -> str:
    resp = client.get("/v1/timestamps")
    assert resp.status_code == 200
    payload = resp.json()
    timestamps = payload.get("timestamps", [])
    if not timestamps:
        pytest.skip("No timestamps available in current dataset")
    return timestamps[0]


def test_healthz(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_datasets(client: TestClient) -> None:
    resp = client.get("/v1/datasets")
    assert resp.status_code == 200
    assert "dataset" in resp.json()


def test_cors_allows_local_dev_origin(client: TestClient) -> None:
    origin = "http://127.0.0.1:5178"
    resp = client.get("/v1/datasets", headers={"Origin": origin})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == origin
    assert resp.headers.get("access-control-allow-credentials") == "true"


def test_layers(client: TestClient) -> None:
    ts = _pick_timestamp(client)
    resp = client.get("/v1/layers", params={"timestamp": ts})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["layers"]
    heat_item = next(item for item in payload["layers"] if item["id"] == "ais_heatmap")
    assert isinstance(heat_item["available"], bool)


def test_timestamps_all_alias_returns_full_set(client: TestClient) -> None:
    full = client.get("/v1/timestamps")
    all_alias = client.get("/v1/timestamps", params={"month": "all"})
    assert full.status_code == 200
    assert all_alias.status_code == 200
    assert full.json().get("timestamps", []) == all_alias.json().get("timestamps", [])


def test_overlay_and_tile_png(client: TestClient) -> None:
    ts = _pick_timestamp(client)
    overlay = client.get(
        "/v1/overlay/ais_heatmap.png",
        params={"timestamp": ts, "bbox": "-180,60,180,86", "size": "800,400"},
    )
    assert overlay.status_code == 200
    assert overlay.headers["content-type"] == "image/png"
    assert len(overlay.content) > 500

    tile = client.get(f"/v1/tiles/ais_heatmap/1/1/0.png", params={"timestamp": ts})
    assert tile.status_code == 200
    assert tile.headers["content-type"] == "image/png"
    assert len(tile.content) > 100


def test_route_plan_and_gallery(client: TestClient) -> None:
    ts = _pick_timestamp(client)
    plan_payload = {
        "timestamp": ts,
        "start": TEST_START,
        "goal": TEST_GOAL,
        "policy": {
            "objective": "shortest_distance_under_safety",
            "blocked_sources": ["bathy", "unet_blocked"],
            "caution_mode": "tie_breaker",
            "corridor_bias": 0.2,
            "smoothing": True,
        },
    }
    plan_resp = client.post("/v1/route/plan", json=plan_payload)
    assert plan_resp.status_code == 200
    plan = plan_resp.json()
    assert plan["gallery_id"]
    assert plan["route_geojson"]["geometry"]["type"] == "LineString"

    gallery_resp = client.get("/v1/gallery/list")
    assert gallery_resp.status_code == 200
    assert isinstance(gallery_resp.json().get("items", []), list)
    item_resp = client.get(f"/v1/gallery/{plan['gallery_id']}")
    assert item_resp.status_code == 200
    item = item_resp.json()
    assert item.get("action", {}).get("type") == "route_plan"
    assert item.get("result", {}).get("status") == "success"
    assert isinstance(item.get("timeline", []), list)

    image_resp = client.get(f"/v1/gallery/{plan['gallery_id']}/image.png")
    assert image_resp.status_code == 200
    assert image_resp.headers["content-type"] == "image/png"
    # Non-trivial preview image should be larger than placeholder 1x1 PNG.
    assert len(image_resp.content) > 500

    one_px_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2NcAoAAAAASUVORK5CYII="
    upload_resp = client.post(
        f"/v1/gallery/{plan['gallery_id']}/image",
        json={"image_base64": one_px_png},
    )
    assert upload_resp.status_code == 204
    image_resp2 = client.get(f"/v1/gallery/{plan['gallery_id']}/image.png")
    assert image_resp2.status_code == 200
    assert image_resp2.content == base64.b64decode(one_px_png)


def test_route_plan_modes(client: TestClient) -> None:
    ts = _pick_timestamp(client)
    payload = {
        "timestamp": ts,
        "start": TEST_START,
        "goal": TEST_GOAL,
        "policy": {
            "objective": "shortest_distance_under_safety",
            "blocked_sources": ["bathy", "unet_blocked"],
            "caution_mode": "minimize",
            "corridor_bias": 0.2,
            "smoothing": True,
            "planner": "dstar_lite",
        },
    }
    resp = client.post("/v1/route/plan", json=payload)
    assert resp.status_code == 200
    explain = resp.json()["explain"]
    assert explain["caution_mode"] == "minimize"
    assert explain["effective_caution_penalty"] > 0
    assert explain["planner"] == "dstar_lite"


def test_latest_plan_fallback(client: TestClient) -> None:
    payload = {
        "date": "2024-10-15",
        "hour": 12,
        "start": TEST_START,
        "goal": TEST_GOAL,
        "policy": {
            "objective": "shortest_distance_under_safety",
            "blocked_sources": ["bathy", "unet_blocked"],
            "caution_mode": "tie_breaker",
            "corridor_bias": 0.2,
            "smoothing": True,
            "planner": "dstar_lite",
        },
    }
    resp = client.post("/v1/latest/plan", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["route_geojson"]["geometry"]["type"] == "LineString"
    assert "resolved" in body
    assert body["resolved"]["source"] in {"local_existing", "remote_snapshot", "nearest_local_fallback"}


def test_latest_progress_endpoint(client: TestClient) -> None:
    progress_id = "progress-missing-smoke"
    missing = client.get("/v1/latest/progress", params={"progress_id": progress_id})
    assert missing.status_code == 200
    payload = missing.json()
    assert payload["exists"] is False
    assert payload["status"] == "not_found"
    assert payload["progress_id"] == progress_id


def test_copernicus_config_endpoints(client: TestClient) -> None:
    set_resp = client.post(
        "/v1/latest/copernicus/config",
        json={
            "username": "demo_user",
            "password": "demo_pass",
            "ice_dataset_id": "ice_ds",
            "wave_dataset_id": "wave_ds",
            "wind_dataset_id": "wind_ds",
        },
    )
    assert set_resp.status_code == 200
    payload = set_resp.json()
    assert payload["ok"] is True

    get_resp = client.get("/v1/latest/copernicus/config")
    assert get_resp.status_code == 200
    cfg = get_resp.json()
    assert "configured" in cfg
    assert cfg["username_set"] is True
    assert cfg["password_set"] is True

    status_resp = client.get("/v1/latest/status", params={"timestamp": "2024-07-01_00"})
    assert status_resp.status_code == 200
    st = status_resp.json()
    assert st["timestamp"] == "2024-07-01_00"
    assert "has_latest_meta" in st


def test_route_plan_is_stable_and_stays_out_of_blocked(client: TestClient) -> None:
    ts_ui = _pick_timestamp(client)
    ts = normalize_timestamp(ts_ui)
    payload = {
        "timestamp": ts_ui,
        "start": TEST_START,
        "goal": TEST_GOAL,
        "policy": {
            "objective": "shortest_distance_under_safety",
            "blocked_sources": ["bathy", "unet_blocked"],
            "caution_mode": "tie_breaker",
            "corridor_bias": 0.2,
            "smoothing": True,
        },
    }

    first = client.post("/v1/route/plan", json=payload)
    second = client.post("/v1/route/plan", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    first_data = first.json()
    second_data = second.json()

    first_coords = first_data["route_geojson"]["geometry"]["coordinates"]
    second_coords = second_data["route_geojson"]["geometry"]["coordinates"]
    assert first_coords == second_coords
    assert first_data["explain"]["distance_km"] == second_data["explain"]["distance_km"]
    assert first_data["explain"]["caution_len_km"] == second_data["explain"]["caution_len_km"]

    settings = get_settings()
    blocked_bathy_path = settings.annotation_pack_root / ts / "blocked_mask.npy"
    assert blocked_bathy_path.exists()
    blocked_bathy = np.load(blocked_bathy_path).astype(np.uint8) > 0

    pred_path = settings.pred_root / "unet_v1" / f"{ts}.npy"
    assert pred_path.exists()
    unet_pred = np.load(pred_path).astype(np.uint8)
    assert unet_pred.shape == blocked_bathy.shape
    fused_blocked = blocked_bathy | (unet_pred == 2)

    geo = load_grid_geo(settings=settings, timestamp=ts, shape=blocked_bathy.shape)
    for lon, lat in first_coords:
        r, c, inside = geo.latlon_to_rc(float(lat), float(lon))
        assert inside
        assert not bool(fused_blocked[r, c])


def test_eval_ais_backtest(client: TestClient) -> None:
    ts = _pick_timestamp(client)
    plan_payload = {
        "timestamp": ts,
        "start": TEST_START,
        "goal": TEST_GOAL,
        "policy": {
            "objective": "shortest_distance_under_safety",
            "blocked_sources": ["bathy", "unet_blocked"],
            "caution_mode": "tie_breaker",
            "corridor_bias": 0.2,
            "smoothing": True,
        },
    }
    plan = client.post("/v1/route/plan", json=plan_payload).json()
    eval_resp = client.post("/v1/eval/ais/backtest", json={"gallery_id": plan["gallery_id"]})
    assert eval_resp.status_code == 200
    metrics = eval_resp.json()["metrics"]
    assert "top10pct_hit_rate" in metrics
    assert "alignment_norm_0_1" in metrics
    assert 0.0 <= metrics["alignment_norm_0_1"] <= 1.0


def test_infer_persists_file(client: TestClient) -> None:
    ts = _pick_timestamp(client)
    infer_resp = client.post(
        "/v1/infer",
        json={"timestamp": ts, "model_version": "unet_v1"},
    )
    assert infer_resp.status_code == 200
    payload = infer_resp.json()
    output_file = Path(payload["output_file"])
    assert output_file.exists()


def test_error_payload_shape(client: TestClient) -> None:
    bad = client.get("/v1/layers", params={"timestamp": "bad-ts"})
    assert bad.status_code == 422
    payload = bad.json()
    assert payload["code"] in {"http_error", "validation_error"}
    assert payload["status"] == 422
    assert isinstance(payload["message"], str)
    assert "detail" in payload
