from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import BaseModel

from app.core.compliance import build_compliance_notices
from app.core.config import get_settings
from app.core.gallery import GalleryService
from app.core.report_template import build_report_template, report_template_to_csv, report_template_to_markdown
from app.core.risk_report import build_risk_report


router = APIRouter(tags=["gallery"])


class GalleryImageUploadRequest(BaseModel):
    image_base64: str


@router.get("/gallery/list")
def list_gallery() -> dict:
    return {"items": GalleryService().list_items()}


@router.get("/gallery/deleted")
def list_deleted_gallery() -> dict:
    return {"items": GalleryService().list_deleted_items()}


@router.get("/gallery/{gallery_id}")
def get_gallery(gallery_id: str) -> dict:
    item = GalleryService().get_item(gallery_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Gallery item not found")
    return item


@router.get("/gallery/{gallery_id}/risk-report")
def get_gallery_risk_report(gallery_id: str) -> dict:
    item = GalleryService().get_item(gallery_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Gallery item not found")
    report = build_risk_report(item)
    report["compliance"] = build_compliance_notices(
        settings=get_settings(),
        context="export",
        timestamp=str(item.get("timestamp", "")),
    )
    return report


@router.get("/gallery/{gallery_id}/report-template", response_model=None)
def get_gallery_report_template(gallery_id: str, format: str = "json"):
    item = GalleryService().get_item(gallery_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Gallery item not found")
    settings = get_settings()
    compliance = build_compliance_notices(
        settings=settings,
        context="export",
        timestamp=str(item.get("timestamp", "")),
    )
    risk_report = build_risk_report(item)
    report = build_report_template(gallery_item=item, risk_report=risk_report, compliance=compliance)

    fmt = format.strip().lower()
    if fmt == "json":
        return report
    if fmt == "csv":
        return Response(
            content=report_template_to_csv(report),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="report_template_{gallery_id}.csv"'},
        )
    if fmt in {"md", "markdown"}:
        return Response(
            content=report_template_to_markdown(report),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="report_template_{gallery_id}.md"'},
        )
    raise HTTPException(status_code=422, detail="format must be one of: json,csv,markdown")


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
def delete_gallery(gallery_id: str, soft_delete: bool = Query(default=True)) -> Response:
    deleted = GalleryService().delete(gallery_id, soft_delete=soft_delete)
    if not deleted:
        raise HTTPException(status_code=404, detail="Gallery item not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/gallery/{gallery_id}/restore")
def restore_gallery(gallery_id: str) -> dict:
    restored = GalleryService().restore(gallery_id)
    if not restored:
        raise HTTPException(status_code=404, detail="Deleted gallery item not found")
    return {"ok": True, "gallery_id": gallery_id}
