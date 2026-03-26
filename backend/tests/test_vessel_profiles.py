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


def test_apply_custom_vessel_policy() -> None:
    policy = {
        "planner": "dstar_lite",
        "risk_mode": "balanced",
        "risk_weight_scale": 1.0,
        "risk_budget": 0.8,
        "confidence_level": 0.9,
        "corridor_bias": 0.2,
        "vessel_profile_id": "custom",
        "custom_vessel": {
            "name": "Demo Hull",
            "polar_category": "B",
            "ice_class": "PC5",
            "draft_m": 9.8,
            "min_safe_depth_m": 16.0,
            "risk_mode": "conservative",
            "risk_weight_scale": 1.4,
            "risk_budget": 0.66,
            "confidence_level": 0.97,
            "corridor_bias_multiplier": 0.5,
            "ice_risk_multiplier": 1.6,
            "max_ice_conc": 0.55,
            "max_ice_thickness_m": 0.9,
        },
    }
    effective, vessel, adjustments = apply_vessel_profile_to_policy(policy)
    assert vessel["id"] == "custom"
    assert vessel["name"] == "Demo Hull"
    assert vessel["polar_category"] == "B"
    assert effective["min_safe_depth_m"] == 16.0
    assert effective["risk_weight_scale"] == 1.4
    assert effective["ice_risk_multiplier"] == 1.6
    assert effective["max_ice_conc"] == 0.55
    assert effective["max_ice_thickness_m"] == 0.9
    assert adjustments["applied_min_safe_depth_m"] == 16.0


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


def test_route_plan_accepts_custom_vessel() -> None:
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
                "vessel_profile_id": "custom",
                "custom_vessel": {
                    "name": "Demo Hull",
                    "polar_category": "B",
                    "ice_class": "PC5",
                    "draft_m": 9.8,
                    "min_safe_depth_m": 16.0,
                    "risk_mode": "conservative",
                    "risk_weight_scale": 1.4,
                    "risk_budget": 0.66,
                    "confidence_level": 0.97,
                    "corridor_bias_multiplier": 0.5,
                    "ice_risk_multiplier": 1.6,
                    "max_ice_conc": 0.55,
                    "max_ice_thickness_m": 0.9,
                },
            },
        },
    )
    assert resp.status_code == 200
    explain = resp.json().get("explain", {})
    vessel = explain.get("vessel_profile", {})
    adjustments = explain.get("vessel_profile_adjustments", {})
    assert str(vessel.get("id")) == "custom"
    assert str(vessel.get("name")) == "Demo Hull"
    assert float(adjustments.get("applied_min_safe_depth_m", 0.0)) == 16.0
    assert str(adjustments.get("polar_category")) == "B"
    assert float(adjustments.get("max_ice_conc", 0.0)) == 0.55
