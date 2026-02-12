from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.active_review import ActiveReviewService


router = APIRouter(tags=["active_review"])


class ActiveReviewDecisionRequest(BaseModel):
    run_id: str
    timestamp: str
    decision: str
    note: str = ""


@router.get("/active/review/runs")
def get_active_review_runs() -> dict:
    return {"runs": ActiveReviewService().list_runs()}


@router.get("/active/review/items")
def get_active_review_items(
    run_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
) -> dict:
    return ActiveReviewService().get_items(run_id=run_id, limit=limit)


@router.post("/active/review/decision")
def post_active_review_decision(payload: ActiveReviewDecisionRequest) -> dict:
    service = ActiveReviewService()
    try:
        return service.save_decision(
            run_id=payload.run_id,
            timestamp=payload.timestamp,
            decision=payload.decision,
            note=payload.note,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

