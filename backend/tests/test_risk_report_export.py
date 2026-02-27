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


def test_gallery_risk_report_contains_risk_and_candidate_sections(client: TestClient) -> None:
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
            "planner": "astar",
            "return_candidates": True,
            "candidate_limit": 3,
        },
    }
    plan_resp = client.post("/v1/route/plan", json=plan_payload)
    assert plan_resp.status_code == 200
    gallery_id = str(plan_resp.json()["gallery_id"])

    report_resp = client.get(f"/v1/gallery/{gallery_id}/risk-report")
    assert report_resp.status_code == 200
    report = report_resp.json()

    assert report["report_version"] == "risk_report_v1"
    assert report["gallery_id"] == gallery_id
    assert "summary" in report
    assert "risk" in report
    assert "strategy" in report
    assert "candidate_comparison" in report
    assert "compliance" in report

    risk = report["risk"]
    assert "high_risk_crossing_ratio" in risk
    assert "avoidance_gain" in risk
    assert "constraint" in risk

    cmp_block = report["candidate_comparison"]
    assert isinstance(cmp_block.get("count"), int)
    assert isinstance(cmp_block.get("items"), list)
    if cmp_block["items"]:
        sample = cmp_block["items"][0]
        assert "distance_km" in sample
        assert "risk_exposure" in sample
        assert "pareto_rank" in sample

    compliance = report["compliance"]
    assert compliance["version"] == "compliance_v1"
    assert compliance["context"] == "export"
    notice_ids = {str(item.get("id")) for item in compliance.get("notices", []) if isinstance(item, dict)}
    assert "research_only" in notice_ids


def test_gallery_risk_report_404(client: TestClient) -> None:
    resp = client.get("/v1/gallery/nonexistentid/risk-report")
    assert resp.status_code == 404
