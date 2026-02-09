from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, Query, Response

from app.core.dataset import get_dataset_service, normalize_timestamp, ui_timestamp


router = APIRouter(tags=["layers"])

_TRANSPARENT_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2NcAoAAAAASUVORK5CYII="
)


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
    # Placeholder overlay so the frontend can integrate before renderer is implemented.
    _ = (layer, timestamp, bbox, size)
    return Response(content=base64.b64decode(_TRANSPARENT_PNG_BASE64), media_type="image/png")


@router.get("/tiles/{layer}/{z}/{x}/{y}.png")
def get_tile(layer: str, z: int, x: int, y: int, timestamp: str = Query(...)) -> Response:
    _ = (layer, z, x, y, timestamp)
    return Response(content=base64.b64decode(_TRANSPARENT_PNG_BASE64), media_type="image/png")

