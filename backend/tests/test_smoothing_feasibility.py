from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.planning import router as planning_router
from app.planning.router import _is_path_feasible, _max_turn_angle_deg, _smooth_cells_los_constrained


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _pick_plannable_timestamp() -> str:
    settings = get_settings()
    ann_root = settings.annotation_pack_root
    if not ann_root.exists():
        pytest.skip("annotation_pack root missing")
    for d in sorted(ann_root.iterdir()):
        if d.is_dir() and (d / "blocked_mask.npy").exists():
            return d.name
    pytest.skip("No timestamp with blocked_mask found in annotation_pack")


def test_path_feasibility_detects_blocked_crossing() -> None:
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[1, 1] = True
    assert _is_path_feasible([(0, 0), (2, 2)], blocked) is False
    assert _is_path_feasible([(0, 0), (0, 2)], blocked) is True


def test_constrained_smoothing_keeps_turn_under_limit_when_possible() -> None:
    blocked = np.zeros((6, 6), dtype=bool)
    raw = [(0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 3)]
    out = _smooth_cells_los_constrained(raw, blocked, max_turn_deg=110.0)
    assert len(out) >= 2
    assert _max_turn_angle_deg(out) <= 110.0 + 1e-6


def test_plan_route_smoothing_fallback_flag_on_infeasible_smooth(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ts = _pick_plannable_timestamp()

    original_checker = planning_router._is_path_feasible
    calls = {"n": 0}

    def _fake_feasible(cells, blocked):
        calls["n"] += 1
        if calls["n"] == 1:
            return False
        return original_checker(cells, blocked)

    monkeypatch.setattr(planning_router, "_is_path_feasible", _fake_feasible)

    payload = {
        "timestamp": ts,
        "start": {"lat": 70.5, "lon": 30.0},
        "goal": {"lat": 72.0, "lon": 150.0},
        "policy": {
            "objective": "shortest_distance_under_safety",
            "blocked_sources": ["bathy", "unet_blocked"],
            "caution_mode": "tie_breaker",
            "corridor_bias": 0.2,
            "smoothing": True,
            "planner": "astar",
        },
    }
    resp = client.post("/v1/route/plan", json=payload)
    assert resp.status_code == 200
    explain = resp.json()["explain"]
    assert explain["smoothing_feasible"] is False
    assert explain["smoothing_fallback_reason"] == "smoothed_path_crosses_blocked"


def test_plan_route_outputs_raw_and_smoothed_paths(client: TestClient) -> None:
    ts = _pick_plannable_timestamp()
    payload = {
        "timestamp": ts,
        "start": {"lat": 70.5, "lon": 30.0},
        "goal": {"lat": 72.0, "lon": 150.0},
        "policy": {
            "objective": "shortest_distance_under_safety",
            "blocked_sources": ["bathy", "unet_blocked"],
            "caution_mode": "tie_breaker",
            "corridor_bias": 0.2,
            "smoothing": True,
            "planner": "any_angle",
        },
    }
    resp = client.post("/v1/route/plan", json=payload)
    if resp.status_code == 422:
        pytest.skip("any_angle no feasible route in current sample")
    assert resp.status_code == 200
    props = resp.json()["route_geojson"]["properties"]
    assert isinstance(props.get("raw_coordinates"), list)
    assert isinstance(props.get("feasible_smoothed_coordinates"), list)
    assert len(props["raw_coordinates"]) >= 2
    assert len(props["feasible_smoothed_coordinates"]) >= 2
    assert "raw_max_turn_deg" in props
    assert "smoothed_max_turn_deg" in props
