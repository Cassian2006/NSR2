from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response

from app.core.dataset import get_dataset_service, normalize_timestamp, ui_timestamp
from app.core.render import parse_bbox, parse_size, render_overlay_png, tile_bbox
from app.core.config import get_settings


router = APIRouter(tags=["layers"])


@router.get("/datasets")
def get_datasets() -> dict:
    service = get_dataset_service()
    return {"dataset": service.datasets_summary()}


@router.get("/timestamps")
def get_timestamps(month: str | None = Query(default=None, examples=["2024-07"])) -> dict:
    service = get_dataset_service()
    items = [ui_timestamp(ts) for ts in service.list_timestamps(month)]
    return {"timestamps": items}


@router.get("/layers")
def get_layers(timestamp: str = Query(...)) -> dict:
    service = get_dataset_service()
    try:
        normalized = normalize_timestamp(timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"timestamp": normalized, "layers": service.list_layers(normalized)}


@router.get("/overlay/{layer}.png")
def get_overlay(layer: str, timestamp: str = Query(...), bbox: str | None = None, size: str | None = None) -> Response:
    settings = get_settings()
    try:
        normalized = normalize_timestamp(timestamp)
        parsed_bbox = parse_bbox(bbox)
        width, height = parse_size(size, fallback_w=1024, fallback_h=768)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    png = render_overlay_png(
        settings=settings,
        timestamp=normalized,
        layer=layer,
        bbox=parsed_bbox,
        width=width,
        height=height,
    )
    return Response(content=png, media_type="image/png")


@router.get("/tiles/{layer}/{z}/{x}/{y}.png")
def get_tile(layer: str, z: int, x: int, y: int, timestamp: str = Query(...)) -> Response:
    settings = get_settings()
    try:
        normalized = normalize_timestamp(timestamp)
        bbox = tile_bbox(z, x, y)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    png = render_overlay_png(
        settings=settings,
        timestamp=normalized,
        layer=layer,
        bbox=bbox,
        width=256,
        height=256,
    )
    return Response(content=png, media_type="image/png")
