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


class RoutePlanRequest(BaseModel):
    timestamp: str
    start: Coord
    goal: Coord
    policy: PlanPolicy = Field(default_factory=PlanPolicy)


class LatestPlanRequest(BaseModel):
    date: str
    hour: int = 12
    force_refresh: bool = False
    progress_id: str | None = None
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
