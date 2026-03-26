from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from app.core.config import get_settings
from app.core.dataset import DatasetService, normalize_timestamp
from app.core.copernicus_live import is_copernicus_configured
from app.core.latest import LatestDataError, get_latest_meta, resolve_latest_timestamp
from app.core.latest_progress import complete_progress, fail_progress, get_progress, start_progress, update_progress
from app.core.latest_runtime import get_runtime_stats, release_slot, try_acquire_slot
from app.core.latest_source_health import get_source_health_snapshot
from app.core.latest_slo import build_latest_slo_snapshot
from app.core.run_snapshot import save_run_snapshot
from app.core.schemas import CopernicusConfigRequest, LatestPlanRequest
from app.core.stormglass_live import is_stormglass_configured
from app.core.vessel_profiles import apply_vessel_profile_to_policy
from app.core.versioning import build_version_snapshot
from app.planning.router import PlanningError, plan_grid_route, plan_grid_route_dynamic


router = APIRouter(tags=["latest"])


def _scientific_rule_summary(*, effective_policy: dict, vessel_profile: dict) -> dict:
    min_safe_depth = effective_policy.get("min_safe_depth_m")
    max_ice_conc = effective_policy.get("max_ice_conc")
    max_ice_thickness_m = effective_policy.get("max_ice_thickness_m")
    return {
        "risk_representation": {
            "primary": "continuous_risk_field",
            "auxiliary": ["unet_zones_outline", "ais_history_display"],
            "hard_constraints": [
                "bathymetry_blocked",
                "shallow_water_threshold",
                "ice_concentration_threshold",
                "ice_thickness_threshold",
            ],
        },
        "ais_role": {
            "in_risk_field": False,
            "corridor_preference_enabled": bool(effective_policy.get("ais_corridor_enabled", False)),
            "note": "AIS is treated as historical corridor preference only, not as safety truth.",
        },
        "vessel_constraints": {
            "profile_id": vessel_profile.get("id"),
            "polar_category": vessel_profile.get("polar_category"),
            "ice_class": vessel_profile.get("ice_class"),
            "draft_m": vessel_profile.get("draft_m"),
            "min_safe_depth_m": float(min_safe_depth) if min_safe_depth is not None else None,
            "max_ice_conc": float(max_ice_conc) if max_ice_conc is not None else None,
            "max_ice_thickness_m": float(max_ice_thickness_m) if max_ice_thickness_m is not None else None,
            "ice_risk_multiplier": float(effective_policy.get("ice_risk_multiplier", 1.0)),
        },
    }


def _build_dynamic_timestamps(*, anchor_timestamp: str, all_timestamps: list[str], window: int) -> list[str]:
    normalized_anchor = normalize_timestamp(anchor_timestamp)
    timeline = sorted({normalize_timestamp(ts) for ts in all_timestamps if str(ts).strip()})
    if normalized_anchor not in timeline:
        timeline.append(normalized_anchor)
        timeline = sorted(set(timeline))

    try:
        anchor_idx = timeline.index(normalized_anchor)
    except ValueError:
        return [normalized_anchor]

    selected: list[str] = [normalized_anchor]
    left = anchor_idx - 1
    right = anchor_idx + 1
    target = max(2, int(window))
    while len(selected) < target and (left >= 0 or right < len(timeline)):
        if right < len(timeline):
            selected.append(timeline[right])
            right += 1
            if len(selected) >= target:
                break
        if left >= 0:
            selected.append(timeline[left])
            left -= 1

    return sorted(set(selected))


def _plan_latest_static_with_fallback(
    *,
    settings,
    policy: dict,
    timestamp: str,
    start: tuple[float, float],
    goal: tuple[float, float],
):
    used_policy = dict(policy)
    fallback_note = None
    try:
        result = plan_grid_route(
            settings=settings,
            timestamp=timestamp,
            start=start,
            goal=goal,
            model_version="unet_v1",
            ais_corridor_enabled=bool(policy.get("ais_corridor_enabled", False)),
            corridor_bias=policy["corridor_bias"],
            caution_mode=policy["caution_mode"],
            min_safe_depth_m=float(policy.get("min_safe_depth_m")) if policy.get("min_safe_depth_m") is not None else None,
            ice_risk_multiplier=float(policy.get("ice_risk_multiplier", 1.0)),
            max_ice_conc=float(policy.get("max_ice_conc")) if policy.get("max_ice_conc") is not None else None,
            max_ice_thickness_m=float(policy.get("max_ice_thickness_m")) if policy.get("max_ice_thickness_m") is not None else None,
            smoothing=policy["smoothing"],
            blocked_sources=policy["blocked_sources"],
            planner=policy["planner"],
            risk_mode=policy["risk_mode"],
            risk_weight_scale=policy["risk_weight_scale"],
            risk_constraint_mode=policy["risk_constraint_mode"],
            risk_budget=policy["risk_budget"],
            confidence_level=policy["confidence_level"],
            uncertainty_uplift=policy["uncertainty_uplift"],
            uncertainty_uplift_scale=policy["uncertainty_uplift_scale"],
        )
    except PlanningError as exc:
        # Live latest snapshots can temporarily make U-Net too conservative.
        # Retry with bathy hard mask only to preserve route usability.
        if "unet_blocked" not in policy["blocked_sources"]:
            raise
        result = plan_grid_route(
            settings=settings,
            timestamp=timestamp,
            start=start,
            goal=goal,
            model_version="unet_v1",
            ais_corridor_enabled=bool(policy.get("ais_corridor_enabled", False)),
            corridor_bias=policy["corridor_bias"],
            caution_mode=policy["caution_mode"],
            min_safe_depth_m=float(policy.get("min_safe_depth_m")) if policy.get("min_safe_depth_m") is not None else None,
            ice_risk_multiplier=float(policy.get("ice_risk_multiplier", 1.0)),
            max_ice_conc=float(policy.get("max_ice_conc")) if policy.get("max_ice_conc") is not None else None,
            max_ice_thickness_m=float(policy.get("max_ice_thickness_m")) if policy.get("max_ice_thickness_m") is not None else None,
            smoothing=policy["smoothing"],
            blocked_sources=["bathy"],
            planner=policy["planner"],
            risk_mode=policy["risk_mode"],
            risk_weight_scale=policy["risk_weight_scale"],
            risk_constraint_mode=policy["risk_constraint_mode"],
            risk_budget=policy["risk_budget"],
            confidence_level=policy["confidence_level"],
            uncertainty_uplift=policy["uncertainty_uplift"],
            uncertainty_uplift_scale=policy["uncertainty_uplift_scale"],
        )
        used_policy["blocked_sources"] = ["bathy"]
        fallback_note = f"fallback_to_bathy_only: {exc}"

    return result, used_policy, fallback_note


@router.post("/latest/plan")
def plan_latest(payload: LatestPlanRequest) -> dict:
    settings = get_settings()
    version_snapshot = build_version_snapshot(settings=settings, model_version="unet_v1")
    run_snapshot_meta = {"snapshot_id": "", "snapshot_file": ""}
    progress_id = (payload.progress_id or f"latest-{uuid4().hex}").strip()
    start_progress(progress_id, message="Starting latest data flow")

    if not try_acquire_slot():
        runtime = get_runtime_stats()
        fail_progress(
            progress_id,
            error=f"latest pipeline is busy ({runtime['active']}/{runtime['max_concurrent']})",
            phase="queue",
        )
        raise HTTPException(
            status_code=429,
            detail=f"latest pipeline is busy ({runtime['active']}/{runtime['max_concurrent']}); please retry shortly",
        )

    try:
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
        except Exception as exc:  # pragma: no cover - defensive
            fail_progress(progress_id, error=str(exc), phase="resolve")
            raise HTTPException(status_code=500, detail="latest resolve failed") from exc

        update_progress(progress_id, phase="plan", message="Planning route", percent=95)
        requested_policy = payload.policy.model_dump()
        effective_policy, vessel_profile, vessel_adjustments = apply_vessel_profile_to_policy(requested_policy)
        used_policy = dict(effective_policy)
        fallback_note = None
        start_coord = (payload.start.lat, payload.start.lon)
        goal_coord = (payload.goal.lat, payload.goal.lon)
        dynamic_meta = {
            "enabled": bool(payload.dynamic_replan_enabled),
            "mode": "disabled",
            "requested_window": int(payload.dynamic_window),
            "requested_advance_steps": int(payload.dynamic_advance_steps),
            "used_timestamps": [latest.timestamp],
            "note": "",
        }

        result = None
        if payload.dynamic_replan_enabled:
            try:
                all_timestamps = DatasetService(settings=settings).list_timestamps(month="all")
                dynamic_timestamps = _build_dynamic_timestamps(
                    anchor_timestamp=latest.timestamp,
                    all_timestamps=all_timestamps,
                    window=payload.dynamic_window,
                )
            except Exception as exc:
                dynamic_timestamps = [latest.timestamp]
                dynamic_meta["note"] = f"timeline_build_failed: {exc}"

            dynamic_meta["used_timestamps"] = dynamic_timestamps
            if len(dynamic_timestamps) >= 2:
                try:
                    update_progress(progress_id, phase="dynamic_plan", message="Running dynamic replanning timeline", percent=96)
                    result = plan_grid_route_dynamic(
                        settings=settings,
                        timestamps=dynamic_timestamps,
                        start=start_coord,
                        goal=goal_coord,
                        model_version="unet_v1",
                        ais_corridor_enabled=bool(effective_policy.get("ais_corridor_enabled", False)),
                        corridor_bias=float(effective_policy["corridor_bias"]),
                        caution_mode=str(effective_policy["caution_mode"]),
                        min_safe_depth_m=float(effective_policy.get("min_safe_depth_m")) if effective_policy.get("min_safe_depth_m") is not None else None,
                        ice_risk_multiplier=float(effective_policy.get("ice_risk_multiplier", 1.0)),
                        max_ice_conc=float(effective_policy.get("max_ice_conc")) if effective_policy.get("max_ice_conc") is not None else None,
                        max_ice_thickness_m=float(effective_policy.get("max_ice_thickness_m")) if effective_policy.get("max_ice_thickness_m") is not None else None,
                        smoothing=bool(effective_policy["smoothing"]),
                        blocked_sources=list(effective_policy["blocked_sources"]),
                        planner=str(effective_policy["planner"]),
                        advance_steps=payload.dynamic_advance_steps,
                        risk_mode=str(effective_policy["risk_mode"]),
                        risk_weight_scale=float(effective_policy["risk_weight_scale"]),
                        risk_constraint_mode=str(effective_policy["risk_constraint_mode"]),
                        risk_budget=float(effective_policy["risk_budget"]),
                        confidence_level=float(effective_policy["confidence_level"]),
                        uncertainty_uplift=bool(effective_policy["uncertainty_uplift"]),
                        uncertainty_uplift_scale=float(effective_policy["uncertainty_uplift_scale"]),
                        dynamic_replan_mode=str(effective_policy["dynamic_replan_mode"]),
                        replan_blocked_ratio=float(effective_policy["replan_blocked_ratio"]),
                        replan_risk_spike=float(effective_policy["replan_risk_spike"]),
                        replan_corridor_min=float(effective_policy["replan_corridor_min"]),
                        replan_max_skip_steps=int(effective_policy["replan_max_skip_steps"]),
                        dynamic_risk_switch_enabled=bool(effective_policy["dynamic_risk_switch_enabled"]),
                        dynamic_risk_budget_km=float(effective_policy["dynamic_risk_budget_km"]),
                        dynamic_risk_warn_ratio=float(effective_policy["dynamic_risk_warn_ratio"]),
                        dynamic_risk_hard_ratio=float(effective_policy["dynamic_risk_hard_ratio"]),
                        dynamic_risk_warn_mode=str(effective_policy["dynamic_risk_warn_mode"]),
                        dynamic_risk_hard_mode=str(effective_policy["dynamic_risk_hard_mode"]),
                        dynamic_risk_switch_min_interval=int(effective_policy["dynamic_risk_switch_min_interval"]),
                    )
                    dynamic_meta["mode"] = "dynamic"
                except PlanningError as exc:
                    dynamic_meta["mode"] = "static_fallback"
                    dynamic_meta["note"] = f"dynamic_planning_failed: {exc}"
            else:
                dynamic_meta["mode"] = "static_fallback"
                dynamic_meta["note"] = "insufficient_timestamps_for_dynamic_replan"

        if result is None:
            try:
                result, used_policy, fallback_note = _plan_latest_static_with_fallback(
                    settings=settings,
                    policy=effective_policy,
                    timestamp=latest.timestamp,
                    start=start_coord,
                    goal=goal_coord,
                )
                if payload.dynamic_replan_enabled and dynamic_meta["mode"] == "disabled":
                    dynamic_meta["mode"] = "static_only"
            except PlanningError as exc:
                fail_progress(progress_id, error=str(exc), phase="plan")
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        explain = dict(result.explain)
        explain["vessel_profile"] = vessel_profile
        explain["vessel_profile_adjustments"] = vessel_adjustments
        explain["policy_requested"] = requested_policy
        explain["policy_effective"] = effective_policy
        explain["scientific_rules_applied"] = _scientific_rule_summary(
            effective_policy=effective_policy,
            vessel_profile=vessel_profile,
        )
        if fallback_note:
            explain["planning_fallback"] = fallback_note
        if payload.dynamic_replan_enabled:
            explain["latest_dynamic"] = dynamic_meta
        try:
            run_snapshot_meta = save_run_snapshot(
                settings=settings,
                kind="latest_plan",
                config={
                    "endpoint": "/v1/latest/plan",
                    "request": {
                        "date": payload.date,
                        "hour": payload.hour,
                        "force_refresh": payload.force_refresh,
                        "start": payload.start.model_dump(),
                        "goal": payload.goal.model_dump(),
                        "policy": requested_policy,
                    },
                },
                result={
                    "resolved_timestamp": latest.timestamp,
                    "resolved_source": latest.source,
                    "distance_km": float(explain.get("distance_km", 0.0)),
                    "caution_len_km": float(explain.get("caution_len_km", 0.0)),
                    "dynamic_mode": str(dynamic_meta.get("mode", "disabled")),
                    "dynamic_timestamps": list(dynamic_meta.get("used_timestamps", [])),
                },
                version_snapshot=version_snapshot,
                replay={
                    "runner": "api.latest_plan",
                    "endpoint": "/v1/latest/plan",
                    "payload": {
                        "date": payload.date,
                        "hour": payload.hour,
                        "force_refresh": payload.force_refresh,
                        "start": payload.start.model_dump(),
                        "goal": payload.goal.model_dump(),
                        "policy": requested_policy,
                    },
                },
                tags=["latest", "planning"],
            )
        except Exception:
            run_snapshot_meta = {"snapshot_id": "", "snapshot_file": ""}
        complete_progress(progress_id, message="Latest prediction and route planning completed")

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
                "dynamic": dynamic_meta,
            },
            "latest_meta": get_latest_meta(settings, latest.timestamp),
            "version_snapshot": version_snapshot,
            "run_snapshot_id": run_snapshot_meta["snapshot_id"],
            "run_snapshot_file": run_snapshot_meta["snapshot_file"],
        }
    finally:
        release_slot()


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


@router.get("/latest/stormglass/config")
def get_stormglass_config() -> dict:
    settings = get_settings()
    return {
        "configured": is_stormglass_configured(settings),
        "api_key_set": bool(settings.stormglass_api_key),
        "source_preference": settings.stormglass_source_preference,
        "sample_lat_count": int(settings.stormglass_sample_lat_count),
        "sample_lon_count": int(settings.stormglass_sample_lon_count),
        "request_timeout_sec": int(settings.stormglass_request_timeout_sec),
        "cache_root": str(settings.stormglass_cache_root),
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


@router.get("/latest/runtime")
def latest_runtime() -> dict:
    return get_runtime_stats()


@router.get("/latest/sources/health")
def latest_sources_health(timestamp: str | None = Query(default=None, description="optional normalized timestamp YYYY-MM-DD_HH")) -> dict:
    settings = get_settings()
    snapshot = get_source_health_snapshot()
    snapshot["slo"] = build_latest_slo_snapshot(settings=settings, timestamp=timestamp)
    return snapshot
