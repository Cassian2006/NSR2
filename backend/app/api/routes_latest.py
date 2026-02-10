from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from app.core.config import get_settings
from app.core.copernicus_live import is_copernicus_configured
from app.core.latest import LatestDataError, get_latest_meta, resolve_latest_timestamp
from app.core.latest_progress import complete_progress, fail_progress, get_progress, start_progress, update_progress
from app.core.schemas import CopernicusConfigRequest, LatestPlanRequest
from app.planning.router import PlanningError, plan_grid_route


router = APIRouter(tags=["latest"])


@router.post("/latest/plan")
def plan_latest(payload: LatestPlanRequest) -> dict:
    settings = get_settings()
    progress_id = (payload.progress_id or f"latest-{uuid4().hex}").strip()
    start_progress(progress_id, message="开始最新数据流程")

    try:
        latest = resolve_latest_timestamp(
            settings=settings,
            date=payload.date,
            hour=payload.hour,
            force_refresh=payload.force_refresh,
            progress_cb=lambda phase, message, percent: update_progress(
                progress_id,
                phase=phase,
                message=message,
                percent=percent,
            ),
        )
    except LatestDataError as exc:
        fail_progress(progress_id, error=str(exc), phase="resolve")
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    update_progress(progress_id, phase="plan", message="正在执行航线规划", percent=95)
    used_policy = payload.policy.model_dump()
    fallback_note = None
    try:
        result = plan_grid_route(
            settings=settings,
            timestamp=latest.timestamp,
            start=(payload.start.lat, payload.start.lon),
            goal=(payload.goal.lat, payload.goal.lon),
            model_version="unet_v1",
            corridor_bias=payload.policy.corridor_bias,
            caution_mode=payload.policy.caution_mode,
            smoothing=payload.policy.smoothing,
            blocked_sources=payload.policy.blocked_sources,
            planner=payload.policy.planner,
        )
    except PlanningError as exc:
        # Live latest snapshots can temporarily make U-Net too conservative.
        # Retry with bathy hard mask only to preserve route usability.
        if "unet_blocked" in payload.policy.blocked_sources:
            try:
                update_progress(progress_id, phase="plan", message="UNet约束过严，切换为仅水深硬约束重试", percent=97)
                result = plan_grid_route(
                    settings=settings,
                    timestamp=latest.timestamp,
                    start=(payload.start.lat, payload.start.lon),
                    goal=(payload.goal.lat, payload.goal.lon),
                    model_version="unet_v1",
                    corridor_bias=payload.policy.corridor_bias,
                    caution_mode=payload.policy.caution_mode,
                    smoothing=payload.policy.smoothing,
                    blocked_sources=["bathy"],
                    planner=payload.policy.planner,
                )
                used_policy["blocked_sources"] = ["bathy"]
                fallback_note = f"fallback_to_bathy_only: {exc}"
            except PlanningError:
                fail_progress(progress_id, error=str(exc), phase="plan")
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        else:
            fail_progress(progress_id, error=str(exc), phase="plan")
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    explain = dict(result.explain)
    if fallback_note:
        explain["planning_fallback"] = fallback_note
    complete_progress(progress_id, message="最新预测与规划完成")

    return {
        "route_geojson": result.route_geojson,
        "explain": explain,
        "progress_id": progress_id,
        "resolved": {
            "requested_date": payload.date,
            "requested_hour": payload.hour,
            "force_refresh": payload.force_refresh,
            "progress_id": progress_id,
            "used_timestamp": latest.timestamp,
            "source": latest.source,
            "note": latest.note,
            "used_policy": used_policy,
        },
        "latest_meta": get_latest_meta(settings, latest.timestamp),
    }


@router.post("/latest/copernicus/config")
def set_copernicus_config(payload: CopernicusConfigRequest) -> dict:
    settings = get_settings()

    if payload.username is not None:
        settings.copernicus_username = payload.username.strip()
    if payload.password is not None:
        settings.copernicus_password = payload.password.strip()
    if payload.ice_dataset_id is not None:
        settings.copernicus_ice_dataset_id = payload.ice_dataset_id.strip()
    if payload.wave_dataset_id is not None:
        settings.copernicus_wave_dataset_id = payload.wave_dataset_id.strip()
    if payload.wind_dataset_id is not None:
        settings.copernicus_wind_dataset_id = payload.wind_dataset_id.strip()
    if payload.ice_var is not None:
        settings.copernicus_ice_var = payload.ice_var.strip()
    if payload.ice_thick_var is not None:
        settings.copernicus_ice_thick_var = payload.ice_thick_var.strip()
    if payload.wave_var is not None:
        settings.copernicus_wave_var = payload.wave_var.strip()
    if payload.wind_u_var is not None:
        settings.copernicus_wind_u_var = payload.wind_u_var.strip()
    if payload.wind_v_var is not None:
        settings.copernicus_wind_v_var = payload.wind_v_var.strip()

    return {
        "ok": True,
        "configured": is_copernicus_configured(settings),
        "datasets": {
            "ice_dataset_id": settings.copernicus_ice_dataset_id,
            "wave_dataset_id": settings.copernicus_wave_dataset_id,
            "wind_dataset_id": settings.copernicus_wind_dataset_id,
        },
        "variables": {
            "ice_var": settings.copernicus_ice_var,
            "ice_thick_var": settings.copernicus_ice_thick_var,
            "wave_var": settings.copernicus_wave_var,
            "wind_u_var": settings.copernicus_wind_u_var,
            "wind_v_var": settings.copernicus_wind_v_var,
        },
    }


@router.get("/latest/copernicus/config")
def get_copernicus_config() -> dict:
    settings = get_settings()
    return {
        "configured": is_copernicus_configured(settings),
        "username_set": bool(settings.copernicus_username),
        "password_set": bool(settings.copernicus_password),
        "datasets": {
            "ice_dataset_id": settings.copernicus_ice_dataset_id,
            "wave_dataset_id": settings.copernicus_wave_dataset_id,
            "wind_dataset_id": settings.copernicus_wind_dataset_id,
        },
        "variables": {
            "ice_var": settings.copernicus_ice_var,
            "ice_thick_var": settings.copernicus_ice_thick_var,
            "wave_var": settings.copernicus_wave_var,
            "wind_u_var": settings.copernicus_wind_u_var,
            "wind_v_var": settings.copernicus_wind_v_var,
        },
    }


@router.get("/latest/status")
def latest_status(timestamp: str = Query(..., description="normalized timestamp YYYY-MM-DD_HH")) -> dict:
    settings = get_settings()
    meta = get_latest_meta(settings=settings, timestamp=timestamp)
    exists = bool(meta)
    return {
        "timestamp": timestamp,
        "has_latest_meta": exists,
        "meta": meta,
    }


@router.get("/latest/progress")
def latest_progress(progress_id: str = Query(..., description="latest plan progress id")) -> dict:
    return get_progress(progress_id)
