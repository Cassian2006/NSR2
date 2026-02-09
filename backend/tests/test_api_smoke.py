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
    assert any(item["id"] == "ais_heatmap" for item in payload["layers"])


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
