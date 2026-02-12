from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.annotation_workspace import AnnotationWorkspaceError, AnnotationWorkspaceService


router = APIRouter(tags=["annotation_workspace"])


class AnnotationPointPayload(BaseModel):
    lat: float
    lon: float


class AnnotationOperationPayload(BaseModel):
    id: str = ""
    mode: Literal["add", "erase"]
    shape: Literal["polygon", "stroke"] = "polygon"
    radius_cells: int = Field(default=2, ge=1, le=40)
    points: list[AnnotationPointPayload] = Field(min_length=2, max_length=2000)


class AnnotationPatchSaveRequest(BaseModel):
    timestamp: str
    operations: list[AnnotationOperationPayload] = Field(default_factory=list, max_length=5000)
    note: str = ""
    author: str = "web"


@router.get("/annotation/workspace/patch")
def get_annotation_patch(timestamp: str = Query(...)) -> dict:
    service = AnnotationWorkspaceService()
    try:
        return service.get_patch(timestamp)
    except AnnotationWorkspaceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/annotation/workspace/patch")
def save_annotation_patch(payload: AnnotationPatchSaveRequest) -> dict:
    service = AnnotationWorkspaceService()
    ops = [
        {
            "id": op.id,
            "mode": op.mode,
            "shape": op.shape,
            "radius_cells": op.radius_cells,
            "points": [{"lat": p.lat, "lon": p.lon} for p in op.points],
        }
        for op in payload.operations
    ]
    try:
        return service.save_patch(
            timestamp_raw=payload.timestamp,
            operations_raw=ops,
            note=payload.note,
            author=payload.author,
        )
    except AnnotationWorkspaceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
