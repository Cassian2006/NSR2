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


def _create_gallery_item(client: TestClient) -> str:
    ts = _pick_timestamp(client)
    payload = {
        "timestamp": ts,
        "start": TEST_START,
        "goal": TEST_GOAL,
        "policy": {
            "objective": "shortest_distance_under_safety",
            "blocked_sources": ["bathy", "unet_blocked"],
            "caution_mode": "tie_breaker",
            "corridor_bias": 0.2,
            "smoothing": True,
            "planner": "astar",
            "return_candidates": True,
            "candidate_limit": 3,
        },
    }
    resp = client.post("/v1/route/plan", json=payload)
    assert resp.status_code == 200
    return str(resp.json()["gallery_id"])


def test_report_template_json_contract(client: TestClient) -> None:
    gallery_id = _create_gallery_item(client)
    resp = client.get(f"/v1/gallery/{gallery_id}/report-template?format=json")
    assert resp.status_code == 200
    payload = resp.json()

    assert payload["template_version"] == "report_template_v1"
    assert payload["gallery_id"] == gallery_id
    for key in ["method", "data", "results", "statistics", "limitations", "reproducibility"]:
        assert key in payload
    assert isinstance(payload["limitations"], list)
    assert "compliance" in payload


def test_report_template_csv_and_markdown_export(client: TestClient) -> None:
    gallery_id = _create_gallery_item(client)

    csv_resp = client.get(f"/v1/gallery/{gallery_id}/report-template?format=csv")
    assert csv_resp.status_code == 200
    assert "text/csv" in (csv_resp.headers.get("content-type") or "")
    assert "section,field,value" in csv_resp.text
    assert "method" in csv_resp.text

    md_resp = client.get(f"/v1/gallery/{gallery_id}/report-template?format=markdown")
    assert md_resp.status_code == 200
    assert "text/markdown" in (md_resp.headers.get("content-type") or "")
    assert "## 方法" in md_resp.text
    assert "## 数据" in md_resp.text


def test_report_template_invalid_format(client: TestClient) -> None:
    gallery_id = _create_gallery_item(client)
    resp = client.get(f"/v1/gallery/{gallery_id}/report-template?format=xml")
    assert resp.status_code == 422
