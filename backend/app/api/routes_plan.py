from __future__ import annotations

from fastapi import APIRouter

from app.core.gallery import GalleryService
from app.core.schemas import RoutePlanRequest
from app.planning.router import plan_simple_route


router = APIRouter(tags=["plan"])


@router.post("/route/plan")
def plan_route(payload: RoutePlanRequest) -> dict:
    result = plan_simple_route(
        start=(payload.start.lat, payload.start.lon),
        goal=(payload.goal.lat, payload.goal.lon),
        corridor_bias=payload.policy.corridor_bias,
        caution_mode=payload.policy.caution_mode,
        smoothing=payload.policy.smoothing,
    )

    gallery_id = GalleryService().create(
        {
            "timestamp": payload.timestamp,
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

