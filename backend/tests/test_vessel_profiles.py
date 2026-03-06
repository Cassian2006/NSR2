from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.vessel_profiles import apply_vessel_profile_to_policy, default_vessel_profile_id, list_vessel_profiles
from app.main import app


def test_vessel_profiles_catalog_has_default() -> None:
    profiles = list_vessel_profiles()
    assert len(profiles) >= 4
    ids = {str(p["id"]) for p in profiles}
    assert default_vessel_profile_id() in ids


def test_apply_vessel_profile_to_policy_arc7() -> None:
    policy = {
        "planner": "dstar_lite",
        "risk_mode": "balanced",
        "risk_weight_scale": 1.0,
        "risk_budget": 1.0,
        "confidence_level": 0.9,
        "corridor_bias": 0.2,
        "vessel_profile_id": "arc7_lng",
    }
    effective, vessel, adjustments = apply_vessel_profile_to_policy(policy)
    assert vessel["id"] == "arc7_lng"
    assert effective["risk_mode"] == "conservative"
    assert effective["risk_weight_scale"] > 1.0
    assert effective["risk_budget"] < 1.0
    assert adjustments["applied_corridor_bias"] <= adjustments["requested_corridor_bias"]


def test_apply_vessel_profile_preserves_explicit_policy_when_requested() -> None:
    policy = {
        "planner": "dstar_lite",
        "risk_mode": "aggressive",
        "risk_weight_scale": 0.8,
        "risk_budget": 0.91,
        "confidence_level": 0.88,
        "corridor_bias": 0.2,
        "vessel_profile_id": "arc7_lng",
    }
    effective, vessel, adjustments = apply_vessel_profile_to_policy(policy, preserve_explicit=True)
    assert vessel["id"] == "arc7_lng"
    assert effective["risk_mode"] == "aggressive"
    assert effective["risk_weight_scale"] == 0.8
    assert effective["risk_budget"] == 0.91
    assert effective["confidence_level"] == 0.88
    assert adjustments["applied_risk_mode"] == "aggressive"


def test_vessels_profile_api() -> None:
    client = TestClient(app)
    resp = client.get("/v1/vessels/profiles")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["default_profile_id"]
    assert isinstance(payload["profiles"], list)
    assert any(str(item.get("id")) == "arc7_lng" for item in payload["profiles"])


def test_route_plan_carries_vessel_profile() -> None:
    client = TestClient(app)
    ts_resp = client.get("/v1/timestamps")
    assert ts_resp.status_code == 200
    timestamps = ts_resp.json().get("timestamps", [])
    if not timestamps:
        return
    ts = timestamps[0]
    resp = client.post(
        "/v1/route/plan",
        json={
            "timestamp": ts,
            "start": {"lat": 70.5, "lon": 30.0},
            "goal": {"lat": 72.0, "lon": 150.0},
            "policy": {
                "objective": "shortest_distance_under_safety",
                "blocked_sources": ["bathy", "unet_blocked"],
                "caution_mode": "tie_breaker",
                "corridor_bias": 0.2,
                "smoothing": True,
                "vessel_profile_id": "ice_cargo_1a",
            },
        },
    )
    assert resp.status_code == 200
    explain = resp.json().get("explain", {})
    vessel = explain.get("vessel_profile", {})
    assert str(vessel.get("id")) == "ice_cargo_1a"
