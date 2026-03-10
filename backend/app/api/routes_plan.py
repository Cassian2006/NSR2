from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.gallery import GalleryService
from app.core.dataset import normalize_timestamp
from app.core.geo import load_grid_geo
from app.core.dynamic_state import build_dynamic_state_sequence, save_dynamic_state_sequence
from app.core.run_snapshot import save_run_snapshot
from app.core.schemas import DynamicRoutePlanRequest, RoutePlanRequest
from app.core.vessel_profiles import apply_vessel_profile_to_policy
from app.core.versioning import build_version_snapshot
from app.planning.router import PlanningError, plan_grid_route, plan_grid_route_dynamic


router = APIRouter(tags=["plan"])

_ROUTE_OVERLAP_THRESHOLD = 0.82
_DIVERSITY_PENALTY_RADIUS = 3
_DIVERSITY_PENALTY_SCALE = 0.3


def _run_single_route_plan(
    *,
    settings,
    timestamp: str,
    start: tuple[float, float],
    goal: tuple[float, float],
    policy: dict,
    diversity_penalty: np.ndarray | None = None,
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
        diversity_penalty=diversity_penalty,
    )


def _load_geo_shape(*, settings, timestamp: str) -> tuple[object, tuple[int, int]]:
    blocked_path = Path(settings.annotation_pack_root) / timestamp / "blocked_mask.npy"
    if not blocked_path.exists():
        raise PlanningError(f"blocked_mask missing for timestamp={timestamp}: {blocked_path}")
    blocked = np.load(blocked_path)
    shape = (int(blocked.shape[0]), int(blocked.shape[1]))
    return load_grid_geo(settings, timestamp=timestamp, shape=shape), shape


def _segment_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    rr = 1 if r0 < r1 else -1
    cc = 1 if c0 < c1 else -1
    err = dr - dc
    out: list[tuple[int, int]] = []
    while True:
        out.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += rr
        if e2 < dr:
            err += dr
            c0 += cc
    return out


def _route_cells(route_geojson: dict, *, geo, shape: tuple[int, int]) -> set[tuple[int, int]]:
    coords = route_geojson.get("geometry", {}).get("coordinates", [])
    if not isinstance(coords, list) or len(coords) < 2:
        return set()
    h, w = shape
    route_cells: set[tuple[int, int]] = set()
    prev_rc: tuple[int, int] | None = None
    for point in coords:
        if not isinstance(point, list | tuple) or len(point) < 2:
            continue
        lon = float(point[0])
        lat = float(point[1])
        rr, cc, _ = geo.latlon_to_rc(lat, lon)
        rr = min(max(int(rr), 0), h - 1)
        cc = min(max(int(cc), 0), w - 1)
        rc = (rr, cc)
        if prev_rc is None:
            route_cells.add(rc)
        else:
            route_cells.update(_segment_cells(prev_rc, rc))
        prev_rc = rc
    return route_cells


def _route_overlap_ratio(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(1, min(len(a), len(b))))


def _build_diversity_penalty(shape: tuple[int, int], selected_routes: list[set[tuple[int, int]]]) -> np.ndarray | None:
    if not selected_routes:
        return None
    h, w = shape
    penalty = np.zeros((h, w), dtype=np.float32)
    radius = _DIVERSITY_PENALTY_RADIUS
    for route_cells in selected_routes:
        for rr, cc in route_cells:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr = rr + dr
                    nc = cc + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    dist = max(abs(dr), abs(dc))
                    if dist > radius:
                        continue
                    weight = _DIVERSITY_PENALTY_SCALE * float(radius + 1 - dist) / float(radius + 1)
                    penalty[nr, nc] += weight
    return np.clip(penalty, 0.0, 1.2)


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
        ("distance_first", "Distance First", {"risk_mode": "aggressive", "caution_mode": "tie_breaker", "corridor_bias": 0.12}),
        ("risk_first", "Risk First", {"risk_mode": "conservative", "caution_mode": "minimize", "corridor_bias": 0.32}),
        ("balanced", "Balanced", {"risk_mode": "balanced", "caution_mode": "tie_breaker", "corridor_bias": 0.22}),
        ("any_angle", "Any Angle", {"planner": "any_angle", "risk_mode": "balanced", "caution_mode": "tie_breaker", "smoothing": False}),
        ("hybrid_safe", "Hybrid Safe", {"planner": "hybrid_astar", "risk_mode": "conservative", "caution_mode": "budget", "corridor_bias": 0.28}),
    ]

    geo, shape = _load_geo_shape(settings=settings, timestamp=timestamp)
    seen: set[tuple] = set()
    candidates: list[dict] = []
    accepted_route_cells: list[set[tuple[int, int]]] = []
    requested_route_cells: set[tuple[int, int]] = set()
    pruned_overlap_count = 0
    attempted_count = 0
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
        attempted_count += 1
        try:
            if strategy == "requested":
                cand_result = base_result
            else:
                diversity_penalty = _build_diversity_penalty(shape, accepted_route_cells)
                cand_result = _run_single_route_plan(
                    settings=settings,
                    timestamp=timestamp,
                    start=start,
                    goal=goal,
                    policy=policy,
                    diversity_penalty=diversity_penalty,
                )
            candidate = _candidate_record(
                strategy=strategy,
                label=label,
                result=cand_result,
                policy=policy,
            )
            route_cells = _route_cells(candidate["route_geojson"], geo=geo, shape=shape)
            if strategy == "requested":
                requested_route_cells = set(route_cells)
                candidate["route_overlap_to_requested"] = 1.0
                candidate["route_overlap_to_selected"] = 0.0
                candidate["route_distinct"] = True
                candidates.append(candidate)
                accepted_route_cells.append(route_cells)
                continue

            overlap_to_requested = _route_overlap_ratio(route_cells, requested_route_cells)
            overlap_to_selected = max((_route_overlap_ratio(route_cells, cells) for cells in accepted_route_cells), default=0.0)
            candidate["route_overlap_to_requested"] = round(float(overlap_to_requested), 6)
            candidate["route_overlap_to_selected"] = round(float(overlap_to_selected), 6)
            candidate["route_distinct"] = bool(overlap_to_selected < _ROUTE_OVERLAP_THRESHOLD)
            if overlap_to_selected >= _ROUTE_OVERLAP_THRESHOLD:
                pruned_overlap_count += 1
                continue
            candidates.append(candidate)
            accepted_route_cells.append(route_cells)
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
                    "route_overlap_to_requested": None,
                    "route_overlap_to_selected": None,
                    "route_distinct": False,
                }
            )

    pareto_summary = _pareto_rank_candidates(candidates)
    pareto_summary["attempted_count"] = int(attempted_count)
    pareto_summary["distinct_count"] = int(sum(1 for c in candidates if c.get("status") == "ok" and c.get("route_distinct")))
    pareto_summary["pruned_overlap_count"] = int(pruned_overlap_count)
    pareto_summary["overlap_threshold"] = float(_ROUTE_OVERLAP_THRESHOLD)
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
    policy_data, vessel_profile, vessel_adjustments = apply_vessel_profile_to_policy(
        requested_policy,
        preserve_explicit=True,
    )
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
    policy_data, vessel_profile, vessel_adjustments = apply_vessel_profile_to_policy(
        requested_policy,
        preserve_explicit=True,
    )

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
