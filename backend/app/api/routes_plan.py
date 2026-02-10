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
            planner=payload.policy.planner,
        )
    except PlanningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    route_coords = result.route_geojson.get("geometry", {}).get("coordinates", [])
    action = {
        "type": "route_plan",
        "timestamp": timestamp,
        "start_input": payload.start.model_dump(),
        "goal_input": payload.goal.model_dump(),
        "policy": payload.policy.model_dump(),
    }
    plan_result = {
        "status": "success",
        "distance_km": result.explain["distance_km"],
        "distance_nm": result.explain["distance_nm"],
        "caution_len_km": result.explain["caution_len_km"],
        "corridor_alignment": result.explain["corridor_alignment"],
        "route_points": len(route_coords) if isinstance(route_coords, list) else 0,
        "raw_points": int(result.explain.get("raw_points", 0)),
        "smoothed_points": int(result.explain.get("smoothed_points", 0)),
        "start_adjusted": bool(result.explain.get("start_adjusted", False)),
        "goal_adjusted": bool(result.explain.get("goal_adjusted", False)),
        "blocked_ratio": float(result.explain.get("blocked_ratio", 0.0)),
    }
    timeline = [
        {"event": "route_plan_requested", "status": "ok"},
        {"event": "route_plan_completed", "status": "ok", "result_status": "success"},
    ]

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
            "action": action,
            "result": plan_result,
            "timeline": timeline,
        }
    )

    return {
        "route_geojson": result.route_geojson,
        "explain": result.explain,
        "gallery_id": gallery_id,
    }
