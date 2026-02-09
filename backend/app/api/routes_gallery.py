from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from app.core.gallery import GalleryService


router = APIRouter(tags=["gallery"])


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


@router.delete("/gallery/{gallery_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_gallery(gallery_id: str) -> Response:
    deleted = GalleryService().delete(gallery_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Gallery item not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)

