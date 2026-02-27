from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query

from app.core.compliance import build_compliance_notices
from app.core.config import get_settings


router = APIRouter(tags=["compliance"])


@router.get("/compliance/notices")
def get_compliance_notices(
    context: Literal["workspace", "export"] = Query(default="workspace"),
    timestamp: str | None = Query(default=None),
) -> dict:
    settings = get_settings()
    return build_compliance_notices(settings=settings, context=context, timestamp=timestamp)
