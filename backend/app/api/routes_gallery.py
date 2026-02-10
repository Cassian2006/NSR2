from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel

from app.core.gallery import GalleryService


router = APIRouter(tags=["gallery"])


class GalleryImageUploadRequest(BaseModel):
    image_base64: str


@router.get("/gallery/list")
def list_gallery() -> dict:
    return {"items": GalleryService().list_items()}


@router.get("/gallery/{gallery_id}")
def get_gallery(gallery_id: str) -> dict:
    item = GalleryService().get_item(gallery_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Gallery item not found")
    return item


@router.get("/gallery/{gallery_id}/image.png")
def get_gallery_image(gallery_id: str) -> Response:
    image_bytes = GalleryService().get_image_bytes(gallery_id)
    if image_bytes is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=image_bytes, media_type="image/png")


@router.post("/gallery/{gallery_id}/image", status_code=status.HTTP_204_NO_CONTENT)
def upload_gallery_image(gallery_id: str, payload: GalleryImageUploadRequest) -> Response:
    encoded = payload.image_base64.strip()
    if "," in encoded and encoded.lower().startswith("data:image/png;base64,"):
        encoded = encoded.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid base64 image payload") from exc

    service = GalleryService()
    try:
        ok = service.set_image_bytes(gallery_id, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not ok:
        raise HTTPException(status_code=404, detail="Gallery item not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/gallery/{gallery_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_gallery(gallery_id: str) -> Response:
    deleted = GalleryService().delete(gallery_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Gallery item not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
