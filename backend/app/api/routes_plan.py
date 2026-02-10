from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.gallery import GalleryService
from app.core.dataset import normalize_timestamp
from app.core.schemas import RoutePlanRequest
from app.planning.router import PlanningError, plan_grid_route


router = APIRouter(tags=["plan"])


@router.post("/route/plan")
def plan_route(payload: RoutePlanRequest) -> dict:
    settings = get_settings()
    try:
        timestamp = normalize_timestamp(payload.timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = plan_grid_route(
            settings=settings,
            timestamp=timestamp,
            start=(payload.start.lat, payload.start.lon),
            goal=(payload.goal.lat, payload.goal.lon),
            model_version="unet_v1",
            corridor_bias=payload.policy.corridor_bias,
            caution_mode=payload.policy.caution_mode,
            smoothing=payload.policy.smoothing,
            blocked_sources=payload.policy.blocked_sources,
        )
    except PlanningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    gallery_id = GalleryService().create(
        {
            "timestamp": timestamp,
            "layers": ["bathy", "ais_heatmap", "unet_pred"],
            "start": payload.start.model_dump(),
            "goal": payload.goal.model_dump(),
            "distance_km": result.explain["distance_km"],
            "caution_len_km": result.explain["caution_len_km"],
            "corridor_bias": payload.policy.corridor_bias,
            "model_version": "unet_v1",
            "route_geojson": result.route_geojson,
            "explain": result.explain,
        }
    )

    return {
        "route_geojson": result.route_geojson,
        "explain": result.explain,
        "gallery_id": gallery_id,
    }
