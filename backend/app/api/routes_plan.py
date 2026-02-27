from __future__ import annotations

from copy import deepcopy

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.gallery import GalleryService
from app.core.dataset import normalize_timestamp
from app.core.dynamic_state import build_dynamic_state_sequence, save_dynamic_state_sequence
from app.core.run_snapshot import save_run_snapshot
from app.core.schemas import DynamicRoutePlanRequest, RoutePlanRequest
from app.core.vessel_profiles import apply_vessel_profile_to_policy
from app.core.versioning import build_version_snapshot
from app.planning.router import PlanningError, plan_grid_route, plan_grid_route_dynamic


router = APIRouter(tags=["plan"])


def _run_single_route_plan(
    *,
    settings,
    timestamp: str,
    start: tuple[float, float],
    goal: tuple[float, float],
    policy: dict,
):
    return plan_grid_route(
        settings=settings,
        timestamp=timestamp,
        start=start,
        goal=goal,
        model_version="unet_v1",
        corridor_bias=float(policy["corridor_bias"]),
        caution_mode=str(policy["caution_mode"]),
        smoothing=bool(policy["smoothing"]),
        blocked_sources=list(policy["blocked_sources"]),
        planner=str(policy["planner"]),
        risk_mode=str(policy["risk_mode"]),
        risk_weight_scale=float(policy["risk_weight_scale"]),
        risk_constraint_mode=str(policy["risk_constraint_mode"]),
        risk_budget=float(policy["risk_budget"]),
        confidence_level=float(policy["confidence_level"]),
        uncertainty_uplift=bool(policy["uncertainty_uplift"]),
        uncertainty_uplift_scale=float(policy["uncertainty_uplift_scale"]),
    )


def _candidate_record(
    *,
    strategy: str,
    label: str,
    result,
    policy: dict,
) -> dict:
    explain = result.explain
    return {
        "id": strategy,
        "label": label,
        "strategy": strategy,
        "status": "ok",
        "distance_km": float(explain.get("distance_km", 0.0)),
        "risk_exposure": float(explain.get("route_cost_risk_extra_km", 0.0)),
        "caution_len_km": float(explain.get("caution_len_km", 0.0)),
        "corridor_score": float(explain.get("corridor_alignment", 0.0)),
        "planner": str(explain.get("planner", policy.get("planner", "astar"))),
        "risk_mode": str(explain.get("risk_mode", policy.get("risk_mode", "balanced"))),
        "caution_mode": str(explain.get("caution_mode", policy.get("caution_mode", "tie_breaker"))),
        "policy": {
            "planner": policy.get("planner"),
            "risk_mode": policy.get("risk_mode"),
            "caution_mode": policy.get("caution_mode"),
            "corridor_bias": float(policy.get("corridor_bias", 0.2)),
            "vessel_profile_id": policy.get("vessel_profile_id"),
        },
        "route_geojson": result.route_geojson,
        "explain": explain,
    }


def _pareto_rank_candidates(candidates: list[dict]) -> dict:
    ok = [c for c in candidates if c.get("status") == "ok"]
    if not ok:
        return {
            "objective_axes": ["distance_km", "risk_exposure"],
            "candidate_count": len(candidates),
            "ok_count": 0,
            "failed_count": len(candidates),
            "frontier_count": 0,
            "ranked_ids": [],
        }

    remaining = list(range(len(ok)))
    rank = 1
    while remaining:
        frontier: list[int] = []
        for i in remaining:
            ai = ok[i]
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                aj = ok[j]
                j_better_or_eq = (
                    float(aj["distance_km"]) <= float(ai["distance_km"])
                    and float(aj["risk_exposure"]) <= float(ai["risk_exposure"])
                )
                j_strict_better = (
                    float(aj["distance_km"]) < float(ai["distance_km"])
                    or float(aj["risk_exposure"]) < float(ai["risk_exposure"])
                )
                if j_better_or_eq and j_strict_better:
                    dominated = True
                    break
            if not dominated:
                frontier.append(i)
        for i in frontier:
            ok[i]["pareto_rank"] = rank
            ok[i]["pareto_frontier"] = rank == 1
        remaining = [i for i in remaining if i not in frontier]
        rank += 1

    dist_vals = [float(c["distance_km"]) for c in ok]
    risk_vals = [float(c["risk_exposure"]) for c in ok]
    caution_vals = [float(c["caution_len_km"]) for c in ok]
    corridor_vals = [float(c["corridor_score"]) for c in ok]

    def _norm(values: list[float], v: float) -> float:
        lo = min(values)
        hi = max(values)
        if hi - lo <= 1e-12:
            return 0.0
        return (v - lo) / (hi - lo)

    for c in ok:
        d = _norm(dist_vals, float(c["distance_km"]))
        r = _norm(risk_vals, float(c["risk_exposure"]))
        ca = _norm(caution_vals, float(c["caution_len_km"]))
        co = _norm(corridor_vals, float(c["corridor_score"]))
        c["pareto_score"] = round(float(d + r + 0.25 * ca - 0.15 * co), 6)

    ok_sorted = sorted(
        ok,
        key=lambda c: (
            int(c.get("pareto_rank", 99)),
            float(c.get("pareto_score", 999.0)),
            float(c["distance_km"]),
        ),
    )
    for idx, c in enumerate(ok_sorted, start=1):
        c["pareto_order"] = idx

    ranked_ids = [str(c["id"]) for c in ok_sorted]
    for c in candidates:
        if c.get("status") != "ok":
            c["pareto_rank"] = None
            c["pareto_frontier"] = False
            c["pareto_score"] = None
            c["pareto_order"] = None

    return {
        "objective_axes": ["distance_km", "risk_exposure"],
        "candidate_count": len(candidates),
        "ok_count": len(ok),
        "failed_count": len(candidates) - len(ok),
        "frontier_count": int(sum(1 for c in ok if c.get("pareto_frontier"))),
        "ranked_ids": ranked_ids,
    }


def _build_route_candidates(
    *,
    settings,
    timestamp: str,
    start: tuple[float, float],
    goal: tuple[float, float],
    base_policy: dict,
    base_result,
    candidate_limit: int,
) -> tuple[list[dict], dict]:
    presets = [
        ("requested", "Requested", {}),
        ("distance_first", "Distance First", {"risk_mode": "aggressive", "caution_mode": "tie_breaker"}),
        ("risk_first", "Risk First", {"risk_mode": "conservative", "caution_mode": "minimize"}),
        ("balanced", "Balanced", {"risk_mode": "balanced", "caution_mode": "tie_breaker"}),
    ]

    seen: set[tuple] = set()
    candidates: list[dict] = []
    for strategy, label, overrides in presets:
        if len(candidates) >= max(1, candidate_limit):
            break
        policy = dict(base_policy)
        policy.update(overrides)
        dedupe_key = (
            str(policy["planner"]),
            str(policy["risk_mode"]),
            str(policy["caution_mode"]),
            float(policy["corridor_bias"]),
            bool(policy["smoothing"]),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        try:
            if strategy == "requested":
                cand_result = base_result
            else:
                cand_result = _run_single_route_plan(
                    settings=settings,
                    timestamp=timestamp,
                    start=start,
                    goal=goal,
                    policy=policy,
                )
            candidates.append(
                _candidate_record(
                    strategy=strategy,
                    label=label,
                    result=cand_result,
                    policy=policy,
                )
            )
        except PlanningError as exc:
            candidates.append(
                {
                    "id": strategy,
                    "label": label,
                    "strategy": strategy,
                    "status": "failed",
                    "error": str(exc),
                    "planner": str(policy["planner"]),
                    "risk_mode": str(policy["risk_mode"]),
                    "caution_mode": str(policy["caution_mode"]),
                    "policy": {
                        "planner": policy.get("planner"),
                        "risk_mode": policy.get("risk_mode"),
                        "caution_mode": policy.get("caution_mode"),
                        "corridor_bias": float(policy.get("corridor_bias", 0.2)),
                        "vessel_profile_id": policy.get("vessel_profile_id"),
                    },
                }
            )

    pareto_summary = _pareto_rank_candidates(candidates)
    return candidates, pareto_summary


@router.post("/route/plan")
def plan_route(payload: RoutePlanRequest) -> dict:
    settings = get_settings()
    version_snapshot = build_version_snapshot(settings=settings, model_version="unet_v1")
    run_snapshot_meta = {"snapshot_id": "", "snapshot_file": ""}
    try:
        timestamp = normalize_timestamp(payload.timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    requested_policy = payload.policy.model_dump()
    policy_data, vessel_profile, vessel_adjustments = apply_vessel_profile_to_policy(requested_policy)
    start_rc = (payload.start.lat, payload.start.lon)
    goal_rc = (payload.goal.lat, payload.goal.lon)
    try:
        result = _run_single_route_plan(
            settings=settings,
            timestamp=timestamp,
            start=start_rc,
            goal=goal_rc,
            policy=policy_data,
        )
    except PlanningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    candidates: list[dict] = []
    pareto_summary: dict | None = None
    if payload.policy.return_candidates:
        candidates, pareto_summary = _build_route_candidates(
            settings=settings,
            timestamp=timestamp,
            start=start_rc,
            goal=goal_rc,
            base_policy=policy_data,
            base_result=result,
            candidate_limit=int(payload.policy.candidate_limit),
        )

    explain = dict(result.explain)
    explain["vessel_profile"] = vessel_profile
    explain["vessel_profile_adjustments"] = vessel_adjustments
    explain["policy_requested"] = requested_policy
    explain["policy_effective"] = policy_data
    if pareto_summary is not None:
        explain["pareto_summary"] = pareto_summary

    route_geojson = deepcopy(result.route_geojson)
    route_geojson.setdefault("properties", {})
    route_geojson["properties"].update(explain)

    route_coords = route_geojson.get("geometry", {}).get("coordinates", [])
    action = {
        "type": "route_plan",
        "timestamp": timestamp,
        "start_input": payload.start.model_dump(),
        "goal_input": payload.goal.model_dump(),
        "policy": policy_data,
        "requested_policy": requested_policy,
    }
    plan_result = {
        "status": "success",
        "distance_km": explain["distance_km"],
        "distance_nm": explain["distance_nm"],
        "caution_len_km": explain["caution_len_km"],
        "corridor_alignment": explain["corridor_alignment"],
        "route_points": len(route_coords) if isinstance(route_coords, list) else 0,
        "raw_points": int(explain.get("raw_points", 0)),
        "smoothed_points": int(explain.get("smoothed_points", 0)),
        "start_adjusted": bool(explain.get("start_adjusted", False)),
        "goal_adjusted": bool(explain.get("goal_adjusted", False)),
        "blocked_ratio": float(explain.get("blocked_ratio", 0.0)),
        "candidate_count": len(candidates),
        "pareto_frontier_count": int(pareto_summary.get("frontier_count", 0)) if pareto_summary else 0,
    }
    timeline = [
        {"event": "route_plan_requested", "status": "ok"},
        {"event": "route_plan_completed", "status": "ok", "result_status": "success"},
    ]
    try:
        run_snapshot_meta = save_run_snapshot(
            settings=settings,
            kind="plan",
            config={
                "endpoint": "/v1/route/plan",
                "request": {
                    "timestamp": timestamp,
                    "start": payload.start.model_dump(),
                    "goal": payload.goal.model_dump(),
                    "policy": requested_policy,
                },
            },
            result={
                "gallery_id": "",
                "distance_km": float(explain["distance_km"]),
                "caution_len_km": float(explain["caution_len_km"]),
                "route_points": int(len(route_coords) if isinstance(route_coords, list) else 0),
                "candidate_count": int(len(candidates)),
            },
            version_snapshot=version_snapshot,
            replay={
                "runner": "api.route_plan",
                "endpoint": "/v1/route/plan",
                "payload": {
                    "timestamp": timestamp,
                    "start": payload.start.model_dump(),
                    "goal": payload.goal.model_dump(),
                    "policy": requested_policy,
                },
            },
            tags=["planning", "route_plan"],
        )
    except Exception:
        run_snapshot_meta = {"snapshot_id": "", "snapshot_file": ""}

    gallery_id = GalleryService().create(
        {
            "timestamp": timestamp,
            "layers": ["bathy", "ais_heatmap", "unet_pred"],
            "start": payload.start.model_dump(),
            "goal": payload.goal.model_dump(),
            "distance_km": explain["distance_km"],
            "caution_len_km": explain["caution_len_km"],
            "corridor_bias": float(policy_data.get("corridor_bias", payload.policy.corridor_bias)),
            "model_version": "unet_v1",
            "dataset_version": version_snapshot["dataset_version"],
            "plan_version": version_snapshot["plan_version"],
            "eval_version": version_snapshot["eval_version"],
            "version_snapshot": version_snapshot,
            "route_geojson": route_geojson,
            "explain": explain,
            "candidates": candidates,
            "pareto_summary": pareto_summary,
            "run_snapshot_id": run_snapshot_meta["snapshot_id"],
            "run_snapshot_file": run_snapshot_meta["snapshot_file"],
            "action": action,
            "result": plan_result,
            "timeline": timeline,
        }
    )

    return {
        "route_geojson": route_geojson,
        "explain": explain,
        "candidates": candidates,
        "gallery_id": gallery_id,
        "version_snapshot": version_snapshot,
        "run_snapshot_id": run_snapshot_meta["snapshot_id"],
        "run_snapshot_file": run_snapshot_meta["snapshot_file"],
    }


@router.post("/route/plan/dynamic")
def plan_route_dynamic(payload: DynamicRoutePlanRequest) -> dict:
    settings = get_settings()
    version_snapshot = build_version_snapshot(settings=settings, model_version="unet_v1")
    run_snapshot_meta = {"snapshot_id": "", "snapshot_file": ""}
    dynamic_state_meta = {"sequence_id": "", "sequence_file": "", "checkpoint_file": ""}
    try:
        timestamps = [normalize_timestamp(ts) for ts in payload.timestamps]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if len(timestamps) < 2:
        raise HTTPException(status_code=422, detail="Dynamic route planning requires at least 2 timestamps.")

    requested_policy = payload.policy.model_dump()
    policy_data, vessel_profile, vessel_adjustments = apply_vessel_profile_to_policy(requested_policy)

    try:
        result = plan_grid_route_dynamic(
            settings=settings,
            timestamps=timestamps,
            start=(payload.start.lat, payload.start.lon),
            goal=(payload.goal.lat, payload.goal.lon),
            model_version="unet_v1",
            corridor_bias=float(policy_data["corridor_bias"]),
            caution_mode=str(policy_data["caution_mode"]),
            smoothing=bool(policy_data["smoothing"]),
            blocked_sources=list(policy_data["blocked_sources"]),
            planner=str(policy_data["planner"]),
            advance_steps=payload.advance_steps,
            risk_mode=str(policy_data["risk_mode"]),
            risk_weight_scale=float(policy_data["risk_weight_scale"]),
            risk_constraint_mode=str(policy_data["risk_constraint_mode"]),
            risk_budget=float(policy_data["risk_budget"]),
            confidence_level=float(policy_data["confidence_level"]),
            uncertainty_uplift=bool(policy_data["uncertainty_uplift"]),
            uncertainty_uplift_scale=float(policy_data["uncertainty_uplift_scale"]),
            dynamic_replan_mode=str(policy_data["dynamic_replan_mode"]),
            replan_blocked_ratio=float(policy_data["replan_blocked_ratio"]),
            replan_risk_spike=float(policy_data["replan_risk_spike"]),
            replan_corridor_min=float(policy_data["replan_corridor_min"]),
            replan_max_skip_steps=int(policy_data["replan_max_skip_steps"]),
            dynamic_risk_switch_enabled=bool(policy_data["dynamic_risk_switch_enabled"]),
            dynamic_risk_budget_km=float(policy_data["dynamic_risk_budget_km"]),
            dynamic_risk_warn_ratio=float(policy_data["dynamic_risk_warn_ratio"]),
            dynamic_risk_hard_ratio=float(policy_data["dynamic_risk_hard_ratio"]),
            dynamic_risk_warn_mode=str(policy_data["dynamic_risk_warn_mode"]),
            dynamic_risk_hard_mode=str(policy_data["dynamic_risk_hard_mode"]),
            dynamic_risk_switch_min_interval=int(policy_data["dynamic_risk_switch_min_interval"]),
        )
    except PlanningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result.explain["vessel_profile"] = vessel_profile
    result.explain["vessel_profile_adjustments"] = vessel_adjustments
    result.explain["policy_requested"] = requested_policy
    result.explain["policy_effective"] = policy_data

    route_coords = result.route_geojson.get("geometry", {}).get("coordinates", [])
    try:
        dynamic_state = build_dynamic_state_sequence(
            settings=settings,
            timestamps=timestamps,
            start=payload.start.model_dump(),
            goal=payload.goal.model_dump(),
            policy=policy_data,
            explain=result.explain,
            model_version="unet_v1",
        )
        dynamic_state_meta = save_dynamic_state_sequence(settings=settings, sequence=dynamic_state)
    except Exception:
        dynamic_state_meta = {"sequence_id": "", "sequence_file": "", "checkpoint_file": ""}
    action = {
        "type": "route_plan_dynamic",
        "timestamps": timestamps,
        "start_input": payload.start.model_dump(),
        "goal_input": payload.goal.model_dump(),
        "advance_steps": payload.advance_steps,
        "policy": policy_data,
        "requested_policy": requested_policy,
    }
    plan_result = {
        "status": "success",
        "distance_km": result.explain["distance_km"],
        "distance_nm": result.explain["distance_nm"],
        "caution_len_km": result.explain["caution_len_km"],
        "corridor_alignment": result.explain["corridor_alignment"],
        "route_points": len(route_coords) if isinstance(route_coords, list) else 0,
        "replan_count": len(result.explain.get("dynamic_replans", [])),
        "executed_edges": int(result.explain.get("executed_edges", 0)),
        "dynamic_state_sequence_id": dynamic_state_meta["sequence_id"],
    }
    timeline = [
        {"event": "dynamic_route_plan_requested", "status": "ok"},
        {
            "event": "dynamic_route_plan_completed",
            "status": "ok",
            "result_status": "success",
            "replan_count": len(result.explain.get("dynamic_replans", [])),
        },
    ]
    try:
        run_snapshot_meta = save_run_snapshot(
            settings=settings,
            kind="plan_dynamic",
            config={
                "endpoint": "/v1/route/plan/dynamic",
                "request": {
                    "timestamps": timestamps,
                    "start": payload.start.model_dump(),
                    "goal": payload.goal.model_dump(),
                    "advance_steps": payload.advance_steps,
                    "policy": requested_policy,
                },
            },
            result={
                "gallery_id": "",
                "distance_km": float(result.explain["distance_km"]),
                "caution_len_km": float(result.explain["caution_len_km"]),
                "route_points": int(len(route_coords) if isinstance(route_coords, list) else 0),
                "replan_count": int(len(result.explain.get("dynamic_replans", []))),
                "dynamic_state_sequence_id": dynamic_state_meta["sequence_id"],
            },
            version_snapshot=version_snapshot,
            replay={
                "runner": "api.route_plan_dynamic",
                "endpoint": "/v1/route/plan/dynamic",
                "payload": {
                    "timestamps": timestamps,
                    "start": payload.start.model_dump(),
                    "goal": payload.goal.model_dump(),
                    "advance_steps": payload.advance_steps,
                    "policy": requested_policy,
                },
            },
            tags=["planning", "dynamic"],
        )
    except Exception:
        run_snapshot_meta = {"snapshot_id": "", "snapshot_file": ""}

    gallery_id = GalleryService().create(
        {
            "timestamp": timestamps[0],
            "layers": ["bathy", "ais_heatmap", "unet_pred"],
            "start": payload.start.model_dump(),
            "goal": payload.goal.model_dump(),
            "distance_km": result.explain["distance_km"],
            "caution_len_km": result.explain["caution_len_km"],
            "corridor_bias": float(policy_data.get("corridor_bias", payload.policy.corridor_bias)),
            "model_version": "unet_v1",
            "dataset_version": version_snapshot["dataset_version"],
            "plan_version": version_snapshot["plan_version"],
            "eval_version": version_snapshot["eval_version"],
            "version_snapshot": version_snapshot,
            "route_geojson": result.route_geojson,
            "explain": result.explain,
            "run_snapshot_id": run_snapshot_meta["snapshot_id"],
            "run_snapshot_file": run_snapshot_meta["snapshot_file"],
            "dynamic_state_sequence_id": dynamic_state_meta["sequence_id"],
            "dynamic_state_sequence_file": dynamic_state_meta["sequence_file"],
            "dynamic_state_checkpoint_file": dynamic_state_meta["checkpoint_file"],
            "action": action,
            "result": plan_result,
            "timeline": timeline,
        }
    )

    return {
        "route_geojson": result.route_geojson,
        "explain": result.explain,
        "gallery_id": gallery_id,
        "version_snapshot": version_snapshot,
        "run_snapshot_id": run_snapshot_meta["snapshot_id"],
        "run_snapshot_file": run_snapshot_meta["snapshot_file"],
        "dynamic_state_sequence_id": dynamic_state_meta["sequence_id"],
        "dynamic_state_sequence_file": dynamic_state_meta["sequence_file"],
        "dynamic_state_checkpoint_file": dynamic_state_meta["checkpoint_file"],
    }
