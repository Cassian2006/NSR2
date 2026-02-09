from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.dataset import normalize_timestamp
from app.core.schemas import InferRequest
from app.model.infer import run_stub_unet_inference


router = APIRouter(tags=["infer"])


@router.post("/infer")
def infer(payload: InferRequest) -> dict:
    settings = get_settings()
    timestamp = normalize_timestamp(payload.timestamp)
    output_path = settings.pred_root / payload.model_version / f"{timestamp}.npy"
    stats = run_stub_unet_inference(output_path)
    return {
        "pred_layer": f"unet_pred/{payload.model_version}",
        "timestamp": timestamp,
        "output_file": str(output_path),
        "stats": stats,
    }

