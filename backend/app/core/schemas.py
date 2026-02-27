from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Coord(BaseModel):
    lat: float
    lon: float


class PlanPolicy(BaseModel):
    objective: str = "shortest_distance_under_safety"
    blocked_sources: list[str] = Field(default_factory=lambda: ["bathy", "unet_blocked"])
    caution_mode: str = "tie_breaker"
    corridor_bias: float = 0.2
    smoothing: bool = True
    planner: str = "astar"
    risk_mode: str = "balanced"
    risk_weight_scale: float = Field(default=1.0, ge=0.0, le=5.0)
    risk_constraint_mode: str = "none"
    risk_budget: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence_level: float = Field(default=0.9, ge=0.5, le=0.999)
    return_candidates: bool = False
    candidate_limit: int = Field(default=3, ge=1, le=6)
    uncertainty_uplift: bool = True
    uncertainty_uplift_scale: float = Field(default=1.0, ge=0.0, le=5.0)
    dynamic_replan_mode: str = "on_event"
    replan_blocked_ratio: float = Field(default=0.002, ge=0.0, le=1.0)
    replan_risk_spike: float = Field(default=0.05, ge=0.0, le=5.0)
    replan_corridor_min: float = Field(default=0.05, ge=0.0, le=1.0)
    replan_max_skip_steps: int = Field(default=2, ge=1, le=100)
    dynamic_risk_switch_enabled: bool = False
    dynamic_risk_budget_km: float = Field(default=1.0, ge=0.0, le=1000.0)
    dynamic_risk_warn_ratio: float = Field(default=0.7, ge=0.0, le=10.0)
    dynamic_risk_hard_ratio: float = Field(default=1.0, ge=0.0, le=10.0)
    dynamic_risk_warn_mode: str = "conservative"
    dynamic_risk_hard_mode: str = "conservative"
    dynamic_risk_switch_min_interval: int = Field(default=1, ge=1, le=500)
    vessel_profile_id: str = "arc7_lng"


class RoutePlanRequest(BaseModel):
    timestamp: str
    start: Coord
    goal: Coord
    policy: PlanPolicy = Field(default_factory=PlanPolicy)


class DynamicRoutePlanRequest(BaseModel):
    timestamps: list[str] = Field(min_length=2)
    start: Coord
    goal: Coord
    advance_steps: int = Field(default=12, ge=1, le=500)
    policy: PlanPolicy = Field(default_factory=PlanPolicy)


class LatestPlanRequest(BaseModel):
    date: str
    hour: int = 12
    force_refresh: bool = False
    progress_id: str | None = None
    dynamic_replan_enabled: bool = False
    dynamic_window: int = Field(default=6, ge=2, le=72)
    dynamic_advance_steps: int = Field(default=12, ge=1, le=500)
    start: Coord
    goal: Coord
    policy: PlanPolicy = Field(default_factory=PlanPolicy)


class CopernicusConfigRequest(BaseModel):
    username: str | None = None
    password: str | None = None
    ice_dataset_id: str | None = None
    wave_dataset_id: str | None = None
    wind_dataset_id: str | None = None
    ice_var: str | None = None
    ice_thick_var: str | None = None
    wave_var: str | None = None
    wind_u_var: str | None = None
    wind_v_var: str | None = None


class InferRequest(BaseModel):
    timestamp: str
    model_version: str = "unet_v1"


class GalleryRecord(BaseModel):
    id: str
    created_at: datetime
    timestamp: str
    layers: list[str]
    start: Coord
    goal: Coord
    distance_km: float
    caution_len_km: float
    corridor_bias: float
    model_version: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
