from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import routes_plan
from app.core.geo import GridGeo
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


def test_route_plan_prunes_high_overlap_candidates(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    lat_axis = [80.0 - 0.2 * i for i in range(40)]
    lon_axis = [10.0 + 2.0 * i for i in range(60)]
    geo = GridGeo(lat_axis=np.asarray(lat_axis, dtype=float), lon_axis=np.asarray(lon_axis, dtype=float))
    shape = (geo.h, geo.w)
    h, w = shape
    monkeypatch.setattr(routes_plan, "_load_geo_shape", lambda **_: (geo, shape))

    def _point(row: int, col: int) -> list[float]:
        lat, lon = geo.rc_to_latlon(min(max(row, 0), h - 1), min(max(col, 0), w - 1))
        return [lon, lat]

    requested_coords = [_point(h // 2, 8), _point(h // 2, w // 2), _point(h // 2, w - 8)]
    duplicate_coords = [_point(h // 2, 8), _point(h // 2, w // 2), _point(h // 2, w - 8)]
    distinct_coords = [_point(max(h // 2 - 10, 0), 8), _point(max(h // 2 - 12, 0), w // 2), _point(max(h // 2 - 10, 0), w - 8)]

    def _result(label: str, coords: list[list[float]], distance_km: float, risk: float) -> SimpleNamespace:
        return SimpleNamespace(
            route_geojson={"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords}, "properties": {}},
            explain={
                "distance_km": distance_km,
                "route_cost_risk_extra_km": risk,
                "caution_len_km": 0.0,
                "corridor_alignment": 0.5,
                "planner": "astar",
                "risk_mode": label,
                "caution_mode": "tie_breaker",
            },
        )

    fake_results = {
        ("aggressive", "tie_breaker", "astar"): _result("aggressive", duplicate_coords, 100.0, 12.0),
        ("conservative", "minimize", "astar"): _result("conservative", distinct_coords, 110.0, 8.0),
        ("balanced", "tie_breaker", "astar"): _result("balanced", duplicate_coords, 101.0, 10.0),
        ("balanced", "tie_breaker", "any_angle"): _result("balanced", distinct_coords, 108.0, 9.0),
        ("conservative", "budget", "hybrid_astar"): _result("conservative", duplicate_coords, 109.0, 7.0),
    }

    def _fake_run_single_route_plan(*, policy: dict, **_: object) -> SimpleNamespace:
        key = (str(policy["risk_mode"]), str(policy["caution_mode"]), str(policy["planner"]))
        return fake_results[key]

    monkeypatch.setattr(routes_plan, "_run_single_route_plan", _fake_run_single_route_plan)

    base_policy = _base_policy()
    base_result = _result("balanced", requested_coords, 105.0, 10.0)
    candidates, summary = routes_plan._build_route_candidates(
        settings=object(),
        timestamp="synthetic",
        start=(70.5, 30.0),
        goal=(72.0, 150.0),
        base_policy=base_policy,
        base_result=base_result,
        candidate_limit=4,
    )

    ok_candidates = [c for c in candidates if c.get("status") == "ok"]
    assert len(ok_candidates) >= 2
    assert any(c.get("id") == "requested" for c in ok_candidates)
    assert summary["pruned_overlap_count"] >= 1
    assert summary["distinct_count"] == sum(1 for c in ok_candidates if c.get("route_distinct"))
    assert all(float(c.get("route_overlap_to_selected", 0.0)) < summary["overlap_threshold"] for c in ok_candidates if c.get("id") != "requested")
