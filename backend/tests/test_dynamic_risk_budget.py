from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.planning.router import _dynamic_risk_stage_and_target_mode


TEST_START = {"lat": 70.5, "lon": 30.0}
TEST_GOAL = {"lat": 72.0, "lon": 150.0}


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _pick_timestamps(client: TestClient, need: int = 4) -> list[str]:
    resp = client.get("/v1/timestamps")
    assert resp.status_code == 200
    items = resp.json().get("timestamps", [])
    if len(items) < need:
        pytest.skip(f"Need at least {need} timestamps for dynamic risk budget tests")
    return items[:need]


def test_dynamic_risk_stage_target_mode_switch_logic() -> None:
    stage, mode, usage = _dynamic_risk_stage_and_target_mode(
        enabled=True,
        cumulative_risk_extra_km=0.9,
        budget_km=1.0,
        base_mode="balanced",
        warn_mode="conservative",
        hard_mode="conservative",
        warn_ratio=0.7,
        hard_ratio=1.0,
    )
    assert stage == "warn"
    assert mode == "conservative"
    assert usage == pytest.approx(0.9)

    stage2, mode2, usage2 = _dynamic_risk_stage_and_target_mode(
        enabled=True,
        cumulative_risk_extra_km=1.3,
        budget_km=1.0,
        base_mode="balanced",
        warn_mode="conservative",
        hard_mode="aggressive",
        warn_ratio=0.7,
        hard_ratio=1.0,
    )
    assert stage2 == "hard"
    assert mode2 == "aggressive"
    assert usage2 == pytest.approx(1.3)


def test_route_plan_dynamic_supports_risk_budget_switch_fields(client: TestClient) -> None:
    timestamps = _pick_timestamps(client, need=4)
    payload = {
        "timestamps": timestamps,
        "start": TEST_START,
        "goal": TEST_GOAL,
        "advance_steps": 8,
        "policy": {
            "objective": "shortest_distance_under_safety",
            "blocked_sources": ["bathy", "unet_blocked"],
            "caution_mode": "tie_breaker",
            "corridor_bias": 0.2,
            "smoothing": True,
            "planner": "dstar_lite",
            "risk_mode": "balanced",
            "risk_weight_scale": 1.0,
            "dynamic_replan_mode": "on_event",
            "dynamic_risk_switch_enabled": True,
            "dynamic_risk_budget_km": 0.1,
            "dynamic_risk_warn_ratio": 0.6,
            "dynamic_risk_hard_ratio": 0.9,
            "dynamic_risk_warn_mode": "conservative",
            "dynamic_risk_hard_mode": "conservative",
            "dynamic_risk_switch_min_interval": 1,
        },
    }
    resp = client.post("/v1/route/plan/dynamic", json=payload)
    assert resp.status_code == 200
    explain = resp.json()["explain"]
    assert explain["dynamic_risk_switch_enabled"] is True
    assert "dynamic_risk_switch_events" in explain
    assert "dynamic_risk_mode_timeline" in explain
    assert "dynamic_risk_budget_usage_final" in explain
    assert "dynamic_risk_budget_protection_steps" in explain
    assert isinstance(explain["dynamic_risk_switch_count"], int)
    assert len(explain["dynamic_risk_mode_timeline"]) == int(explain.get("dynamic_execution_steps", 0))
