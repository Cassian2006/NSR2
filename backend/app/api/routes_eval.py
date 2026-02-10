from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.dataset import normalize_timestamp
from app.core.gallery import GalleryService
from app.eval.compare_ais import EvalError, evaluate_route_vs_ais_heatmap


router = APIRouter(tags=["eval"])


class AisBacktestRequest(BaseModel):
    timestamp: str | None = None
    gallery_id: str | None = None
    route_geojson: dict | None = None
    topk_note: str | None = Field(default=None, description="Optional note for tracking experiment context.")


@router.post("/eval/ais/backtest")
def run_ais_backtest(payload: AisBacktestRequest) -> dict:
    route_geojson = payload.route_geojson
    timestamp = payload.timestamp

    if payload.gallery_id:
        item = GalleryService().get_item(payload.gallery_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Gallery item not found")
        if route_geojson is None:
            route_geojson = item.get("route_geojson")
        if timestamp is None:
            timestamp = item.get("timestamp")

    if route_geojson is None:
        raise HTTPException(status_code=422, detail="route_geojson is required (or provide gallery_id)")
    if not timestamp:
        raise HTTPException(status_code=422, detail="timestamp is required (or provide gallery_id)")

    try:
        normalized_ts = normalize_timestamp(timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        metrics = evaluate_route_vs_ais_heatmap(
            settings=get_settings(),
            timestamp=normalized_ts,
            route_geojson=route_geojson,
        )
    except EvalError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "metrics": metrics,
        "gallery_id": payload.gallery_id,
        "note": payload.topk_note or "",
    }

