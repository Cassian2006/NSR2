from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VesselProfile:
    id: str
    name: str
    category: str
    description: str
    polar_category: str
    ice_class: str
    draft_m: float
    min_safe_depth_m: float
    risk_mode: str
    risk_weight_scale: float
    risk_budget: float
    confidence_level: float
    corridor_bias_multiplier: float
    ice_risk_multiplier: float
    max_ice_conc: float
    max_ice_thickness_m: float

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "polar_category": self.polar_category,
            "ice_class": self.ice_class,
            "draft_m": float(self.draft_m),
            "min_safe_depth_m": float(self.min_safe_depth_m),
            "default_policy": {
                "risk_mode": self.risk_mode,
                "risk_weight_scale": float(self.risk_weight_scale),
                "risk_budget": float(self.risk_budget),
                "confidence_level": float(self.confidence_level),
                "corridor_bias_multiplier": float(self.corridor_bias_multiplier),
                "ice_risk_multiplier": float(self.ice_risk_multiplier),
                "max_ice_conc": float(self.max_ice_conc),
                "max_ice_thickness_m": float(self.max_ice_thickness_m),
            },
        }


_PROFILES: tuple[VesselProfile, ...] = (
    VesselProfile(
        id="arc7_lng",
        name="Arc7 LNG Carrier",
        category="commercial",
        description="High ice-class LNG carrier; prioritize safety and robust routing in heavy ice.",
        polar_category="A",
        ice_class="Arc7",
        draft_m=11.5,
        min_safe_depth_m=14.0,
        risk_mode="conservative",
        risk_weight_scale=1.25,
        risk_budget=0.72,
        confidence_level=0.95,
        corridor_bias_multiplier=0.90,
        ice_risk_multiplier=0.85,
        max_ice_conc=0.85,
        max_ice_thickness_m=1.8,
    ),
    VesselProfile(
        id="polar_research",
        name="Polar Research Vessel",
        category="research",
        description="Research vessel profile balancing mission flexibility and operational safety.",
        polar_category="B",
        ice_class="PC6",
        draft_m=8.0,
        min_safe_depth_m=10.0,
        risk_mode="balanced",
        risk_weight_scale=1.10,
        risk_budget=0.82,
        confidence_level=0.93,
        corridor_bias_multiplier=1.00,
        ice_risk_multiplier=1.00,
        max_ice_conc=0.65,
        max_ice_thickness_m=1.1,
    ),
    VesselProfile(
        id="icebreaker_escort",
        name="Icebreaker Escort",
        category="service",
        description="Escort icebreaker profile with higher maneuverability and efficiency-oriented routing.",
        polar_category="A",
        ice_class="PC3",
        draft_m=9.2,
        min_safe_depth_m=11.0,
        risk_mode="balanced",
        risk_weight_scale=0.95,
        risk_budget=0.90,
        confidence_level=0.90,
        corridor_bias_multiplier=0.85,
        ice_risk_multiplier=0.75,
        max_ice_conc=0.95,
        max_ice_thickness_m=2.2,
    ),
    VesselProfile(
        id="ice_cargo_1a",
        name="Ice-class Cargo (1A/1AS)",
        category="commercial",
        description="Typical 1A/1AS cargo vessel with limited ice capability; use conservative routing.",
        polar_category="C",
        ice_class="1A/1AS",
        draft_m=10.2,
        min_safe_depth_m=12.5,
        risk_mode="conservative",
        risk_weight_scale=1.35,
        risk_budget=0.68,
        confidence_level=0.96,
        corridor_bias_multiplier=1.05,
        ice_risk_multiplier=1.25,
        max_ice_conc=0.35,
        max_ice_thickness_m=0.45,
    ),
)


def _coerce_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    return out


def _build_custom_profile(payload: dict[str, Any] | None) -> VesselProfile | None:
    if not isinstance(payload, dict):
        return None
    return VesselProfile(
        id="custom",
        name=str(payload.get("name") or "Custom Vessel"),
        category="custom",
        description="User-defined vessel parameters applied at planning time.",
        polar_category=str(payload.get("polar_category") or "Custom"),
        ice_class=str(payload.get("ice_class") or "Custom"),
        draft_m=max(0.0, _coerce_float(payload.get("draft_m"), 10.0)),
        min_safe_depth_m=max(0.0, _coerce_float(payload.get("min_safe_depth_m"), 12.0)),
        risk_mode=str(payload.get("risk_mode") or "balanced"),
        risk_weight_scale=max(0.0, _coerce_float(payload.get("risk_weight_scale"), 1.0)),
        risk_budget=min(1.0, max(0.0, _coerce_float(payload.get("risk_budget"), 0.8))),
        confidence_level=min(0.999, max(0.5, _coerce_float(payload.get("confidence_level"), 0.9))),
        corridor_bias_multiplier=min(2.0, max(0.0, _coerce_float(payload.get("corridor_bias_multiplier"), 1.0))),
        ice_risk_multiplier=min(3.0, max(0.0, _coerce_float(payload.get("ice_risk_multiplier"), 1.0))),
        max_ice_conc=min(1.0, max(0.0, _coerce_float(payload.get("max_ice_conc"), 0.5))),
        max_ice_thickness_m=min(10.0, max(0.0, _coerce_float(payload.get("max_ice_thickness_m"), 1.0))),
    )


def list_vessel_profiles() -> list[dict[str, Any]]:
    return [profile.to_json() for profile in _PROFILES]


def default_vessel_profile_id() -> str:
    return _PROFILES[0].id


def get_vessel_profile(profile_id: str | None, *, custom_vessel: dict[str, Any] | None = None) -> VesselProfile:
    normalized = str(profile_id or "").strip().lower()
    if normalized == "custom":
        custom = _build_custom_profile(custom_vessel)
        if custom is not None:
            return custom
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
    custom_vessel = effective.get("custom_vessel")
    profile = get_vessel_profile(effective.get("vessel_profile_id"), custom_vessel=custom_vessel if isinstance(custom_vessel, dict) else None)

    effective["ais_corridor_enabled"] = bool(effective.get("ais_corridor_enabled", False))
    requested_corridor_bias = float(effective.get("corridor_bias", 0.2))
    effective_corridor_bias = max(0.0, min(1.0, requested_corridor_bias * float(profile.corridor_bias_multiplier)))

    effective["vessel_profile_id"] = profile.id
    effective["corridor_bias"] = effective_corridor_bias
    effective["min_safe_depth_m"] = float(profile.min_safe_depth_m)
    effective["draft_m"] = float(profile.draft_m)
    effective["ice_class"] = str(profile.ice_class)
    effective["polar_category"] = str(profile.polar_category)
    effective["ice_risk_multiplier"] = float(profile.ice_risk_multiplier)
    effective["max_ice_conc"] = float(profile.max_ice_conc)
    effective["max_ice_thickness_m"] = float(profile.max_ice_thickness_m)

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
        "ais_corridor_enabled": bool(effective.get("ais_corridor_enabled", False)),
        "applied_risk_mode": str(effective.get("risk_mode", profile.risk_mode)),
        "applied_risk_weight_scale": float(effective.get("risk_weight_scale", profile.risk_weight_scale)),
        "applied_risk_budget": float(effective.get("risk_budget", profile.risk_budget)),
        "applied_confidence_level": float(effective.get("confidence_level", profile.confidence_level)),
        "applied_min_safe_depth_m": float(profile.min_safe_depth_m),
        "applied_draft_m": float(profile.draft_m),
        "ice_class": str(profile.ice_class),
        "polar_category": str(profile.polar_category),
        "ice_risk_multiplier": float(profile.ice_risk_multiplier),
        "max_ice_conc": float(profile.max_ice_conc),
        "max_ice_thickness_m": float(profile.max_ice_thickness_m),
    }
    return effective, profile.to_json(), adjustments
