from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


TEST_START = {"lat": 70.5, "lon": 30.0}
TEST_GOAL = {"lat": 72.0, "lon": 150.0}


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _pick_timestamp(client: TestClient) -> str:
    resp = client.get("/v1/timestamps")
    assert resp.status_code == 200
    timestamps = resp.json().get("timestamps", [])
    if not timestamps:
        pytest.skip("No timestamps available in current dataset")
    return str(timestamps[0])


def _assert_version_snapshot(snapshot: dict) -> None:
    assert isinstance(snapshot, dict)
    for key in ("dataset_version", "model_version", "plan_version", "eval_version"):
        assert key in snapshot
        assert isinstance(snapshot[key], str)
        assert snapshot[key].strip() != ""


def test_route_plan_and_gallery_version_fields(client: TestClient) -> None:
    ts = _pick_timestamp(client)
    plan_resp = client.post(
        "/v1/route/plan",
        json={
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
        },
    )
    assert plan_resp.status_code == 200
    plan = plan_resp.json()
    _assert_version_snapshot(plan["version_snapshot"])
    assert plan["version_snapshot"]["model_version"] == "unet_v1"

    gallery_id = plan["gallery_id"]
    item_resp = client.get(f"/v1/gallery/{gallery_id}")
    assert item_resp.status_code == 200
    item = item_resp.json()
    _assert_version_snapshot(item["version_snapshot"])
    assert item["dataset_version"] == item["version_snapshot"]["dataset_version"]
    assert item["model_version"] == item["version_snapshot"]["model_version"]
    assert item["plan_version"] == item["version_snapshot"]["plan_version"]
    assert item["eval_version"] == item["version_snapshot"]["eval_version"]


def test_infer_response_contains_version_snapshot(client: TestClient) -> None:
    ts = _pick_timestamp(client)
    resp = client.post("/v1/infer", json={"timestamp": ts, "model_version": "unet_v1"})
    assert resp.status_code == 200
    body = resp.json()
    _assert_version_snapshot(body["version_snapshot"])
    assert body["version_snapshot"]["model_version"] == "unet_v1"


def test_latest_plan_contains_version_snapshot(client: TestClient) -> None:
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
    _assert_version_snapshot(body["version_snapshot"])
    assert body["version_snapshot"]["model_version"] == "unet_v1"


def test_eval_backtest_contains_version_snapshot(client: TestClient) -> None:
    ts = _pick_timestamp(client)
    plan_resp = client.post(
        "/v1/route/plan",
        json={
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
        },
    )
    assert plan_resp.status_code == 200
    gallery_id = plan_resp.json()["gallery_id"]

    eval_resp = client.post("/v1/eval/ais/backtest", json={"gallery_id": gallery_id})
    assert eval_resp.status_code == 200
    body = eval_resp.json()
    _assert_version_snapshot(body["version_snapshot"])

