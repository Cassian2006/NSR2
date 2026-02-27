from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


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


def _base_policy() -> dict:
    return {
        "objective": "shortest_distance_under_safety",
        "blocked_sources": ["bathy", "unet_blocked"],
        "caution_mode": "tie_breaker",
        "corridor_bias": 0.2,
        "smoothing": True,
        "planner": "astar",
        "risk_mode": "balanced",
        "risk_weight_scale": 1.0,
        "risk_constraint_mode": "none",
        "risk_budget": 1.0,
        "confidence_level": 0.9,
        "return_candidates": True,
        "candidate_limit": 3,
    }


def test_route_plan_returns_candidates_and_pareto(client: TestClient) -> None:
    ts = _pick_timestamp(client)
    payload = {
        "timestamp": ts,
        "start": {"lat": 70.5, "lon": 30.0},
        "goal": {"lat": 72.0, "lon": 150.0},
        "policy": _base_policy(),
    }
    resp = client.post("/v1/route/plan", json=payload)
    assert resp.status_code == 200
    body = resp.json()

    assert "candidates" in body
    candidates = body["candidates"]
    assert isinstance(candidates, list)
    assert len(candidates) >= 1

    ok_items = [c for c in candidates if c.get("status") == "ok"]
    assert ok_items
    for c in ok_items:
        assert "distance_km" in c
        assert "risk_exposure" in c
        assert "caution_len_km" in c
        assert "corridor_score" in c
        assert "pareto_rank" in c
        assert "pareto_frontier" in c

    assert any(bool(c.get("pareto_frontier")) for c in ok_items)
    pareto = body["explain"].get("pareto_summary")
    assert isinstance(pareto, dict)
    assert pareto["candidate_count"] == len(candidates)
    assert pareto["frontier_count"] >= 1


def test_route_plan_candidate_limit_respected(client: TestClient) -> None:
    ts = _pick_timestamp(client)
    policy = _base_policy()
    policy["candidate_limit"] = 2
    payload = {
        "timestamp": ts,
        "start": {"lat": 70.5, "lon": 30.0},
        "goal": {"lat": 72.0, "lon": 150.0},
        "policy": policy,
    }
    resp = client.post("/v1/route/plan", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    candidates = body["candidates"]
    assert len(candidates) <= 2
    assert any(c.get("id") == "requested" for c in candidates)
