from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


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


def test_layers(client: TestClient) -> None:
    ts = _pick_timestamp(client)
    resp = client.get("/v1/layers", params={"timestamp": ts})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["layers"]
    heat_item = next(item for item in payload["layers"] if item["id"] == "ais_heatmap")
    assert isinstance(heat_item["available"], bool)


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
        "start": {"lat": 78.2467, "lon": 15.4650},
        "goal": {"lat": 81.5074, "lon": 58.3811},
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

    image_resp = client.get(f"/v1/gallery/{plan['gallery_id']}/image.png")
    assert image_resp.status_code == 200
    assert image_resp.headers["content-type"] == "image/png"
    # Non-trivial preview image should be larger than placeholder 1x1 PNG.
    assert len(image_resp.content) > 500


def test_route_plan_modes(client: TestClient) -> None:
    ts = _pick_timestamp(client)
    payload = {
        "timestamp": ts,
        "start": {"lat": 78.2467, "lon": 15.4650},
        "goal": {"lat": 81.5074, "lon": 58.3811},
        "policy": {
            "objective": "shortest_distance_under_safety",
            "blocked_sources": ["bathy", "unet_blocked"],
            "caution_mode": "minimize",
            "corridor_bias": 0.2,
            "smoothing": True,
        },
    }
    resp = client.post("/v1/route/plan", json=payload)
    assert resp.status_code == 200
    explain = resp.json()["explain"]
    assert explain["caution_mode"] == "minimize"
    assert explain["effective_caution_penalty"] > 0


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
