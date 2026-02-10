from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.dataset import normalize_timestamp
from app.core.schemas import InferRequest
from app.model.infer import InferenceError, run_unet_inference


router = APIRouter(tags=["infer"])


@router.post("/infer")
def infer(payload: InferRequest) -> dict:
    settings = get_settings()
    try:
        timestamp = normalize_timestamp(payload.timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    output_path = settings.pred_root / payload.model_version / f"{timestamp}.npy"
    try:
        stats = run_unet_inference(
            settings=settings,
            timestamp=timestamp,
            model_version=payload.model_version,
            output_path=output_path,
        )
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "pred_layer": f"unet_pred/{payload.model_version}",
        "timestamp": timestamp,
        "output_file": str(output_path),
        "stats": stats,
    }
