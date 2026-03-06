from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VesselProfile:
    id: str
    name: str
    category: str
    description: str
    ice_class: str
    draft_m: float
    min_safe_depth_m: float
    risk_mode: str
    risk_weight_scale: float
    risk_budget: float
    confidence_level: float
    corridor_bias_multiplier: float

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "ice_class": self.ice_class,
            "draft_m": float(self.draft_m),
            "min_safe_depth_m": float(self.min_safe_depth_m),
            "default_policy": {
                "risk_mode": self.risk_mode,
                "risk_weight_scale": float(self.risk_weight_scale),
                "risk_budget": float(self.risk_budget),
                "confidence_level": float(self.confidence_level),
                "corridor_bias_multiplier": float(self.corridor_bias_multiplier),
            },
        }


_PROFILES: tuple[VesselProfile, ...] = (
    VesselProfile(
        id="arc7_lng",
        name="Arc7 LNG Carrier",
        category="commercial",
        description="High ice-class LNG carrier; prioritize safety and robust routing in heavy ice.",
        ice_class="Arc7",
        draft_m=11.5,
        min_safe_depth_m=14.0,
        risk_mode="conservative",
        risk_weight_scale=1.25,
        risk_budget=0.72,
        confidence_level=0.95,
        corridor_bias_multiplier=0.90,
    ),
    VesselProfile(
        id="polar_research",
        name="Polar Research Vessel",
        category="research",
        description="Research vessel profile balancing mission flexibility and operational safety.",
        ice_class="PC6",
        draft_m=8.0,
        min_safe_depth_m=10.0,
        risk_mode="balanced",
        risk_weight_scale=1.10,
        risk_budget=0.82,
        confidence_level=0.93,
        corridor_bias_multiplier=1.00,
    ),
    VesselProfile(
        id="icebreaker_escort",
        name="Icebreaker Escort",
        category="service",
        description="Escort icebreaker profile with higher maneuverability and efficiency-oriented routing.",
        ice_class="PC3",
        draft_m=9.2,
        min_safe_depth_m=11.0,
        risk_mode="balanced",
        risk_weight_scale=0.95,
        risk_budget=0.90,
        confidence_level=0.90,
        corridor_bias_multiplier=0.85,
    ),
    VesselProfile(
        id="ice_cargo_1a",
        name="Ice-class Cargo (1A/1AS)",
        category="commercial",
        description="Typical 1A/1AS cargo vessel with limited ice capability; use conservative routing.",
        ice_class="1A/1AS",
        draft_m=10.2,
        min_safe_depth_m=12.5,
        risk_mode="conservative",
        risk_weight_scale=1.35,
        risk_budget=0.68,
        confidence_level=0.96,
        corridor_bias_multiplier=1.05,
    ),
)


def list_vessel_profiles() -> list[dict[str, Any]]:
    return [profile.to_json() for profile in _PROFILES]



def default_vessel_profile_id() -> str:
    return _PROFILES[0].id



def get_vessel_profile(profile_id: str | None) -> VesselProfile:
    normalized = str(profile_id or "").strip().lower()
    for profile in _PROFILES:
        if profile.id == normalized:
            return profile
    return _PROFILES[0]



def apply_vessel_profile_to_policy(
    policy: dict[str, Any],
    *,
    preserve_explicit: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    effective = dict(policy)
    profile = get_vessel_profile(effective.get("vessel_profile_id"))
    requested_corridor_bias = float(effective.get("corridor_bias", 0.2))
    effective_corridor_bias = max(0.0, min(1.0, requested_corridor_bias * float(profile.corridor_bias_multiplier)))

    effective["vessel_profile_id"] = profile.id
    effective["corridor_bias"] = effective_corridor_bias

    policy_defaults = {
        "risk_mode": profile.risk_mode,
        "risk_weight_scale": float(profile.risk_weight_scale),
        "risk_budget": float(profile.risk_budget),
        "confidence_level": float(profile.confidence_level),
    }
    for key, value in policy_defaults.items():
        if preserve_explicit and key in policy:
            continue
        effective[key] = value

    adjustments = {
        "requested_corridor_bias": requested_corridor_bias,
        "applied_corridor_bias": effective_corridor_bias,
        "applied_risk_mode": str(effective.get("risk_mode", profile.risk_mode)),
        "applied_risk_weight_scale": float(effective.get("risk_weight_scale", profile.risk_weight_scale)),
        "applied_risk_budget": float(effective.get("risk_budget", profile.risk_budget)),
        "applied_confidence_level": float(effective.get("confidence_level", profile.confidence_level)),
    }
    return effective, profile.to_json(), adjustments
