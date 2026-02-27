from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.core.config import Settings
from app.core.versioning import build_version_snapshot


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_root(settings: Settings) -> Path:
    root = settings.dynamic_state_root
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def build_dynamic_state_version_snapshot(*, settings: Settings, model_version: str) -> dict[str, str]:
    base = build_version_snapshot(settings=settings, model_version=model_version)
    base["state_version"] = str(settings.dynamic_state_version or "state_v1").strip() or "state_v1"
    return base


def _sequence_id(*, timestamps: list[str], policy: dict[str, Any], start: dict[str, float], goal: dict[str, float]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    hint = {
        "timestamps": timestamps,
        "planner": policy.get("planner"),
        "risk_mode": policy.get("risk_mode"),
        "start": start,
        "goal": goal,
    }
    raw = json.dumps(hint, ensure_ascii=False, sort_keys=True).encode("utf-8")
    suffix = hashlib.sha1(raw).hexdigest()[:10]
    return f"dynamic_state_{stamp}_{suffix}"


def _step_record(
    *,
    step_index: int,
    timestamp: str,
    replan_item: dict[str, Any],
    explain: dict[str, Any],
    cumulative_distance_km: float,
    cumulative_runtime_ms: float,
) -> dict[str, Any]:
    planner_name = str(explain.get("planner", "unknown"))
    risk_mode = str(explain.get("risk_mode", "balanced"))
    risk_layer = str(explain.get("risk_layer", "risk_mean"))

    moved_distance_km = _safe_float(replan_item.get("moved_distance_km"), 0.0)
    runtime_ms = _safe_float(replan_item.get("runtime_ms"), 0.0)
    cumulative_distance_km += moved_distance_km
    cumulative_runtime_ms += runtime_ms

    return {
        "step_index": int(step_index),
        "timestamp": str(timestamp),
        "env_state": {
            "timestamp": str(timestamp),
            "grid_family": "aligned_h_w",
        },
        "risk_state": {
            "risk_mode": risk_mode,
            "risk_layer": risk_layer,
            "risk_penalty_mean": round(_safe_float(explain.get("risk_penalty_mean"), 0.0), 6),
            "risk_penalty_p90": round(_safe_float(explain.get("risk_penalty_p90"), 0.0), 6),
            "risk_constraint_mode": str(explain.get("risk_constraint_mode", "none")),
            "risk_constraint_metric_name": str(explain.get("risk_constraint_metric_name", "none")),
            "risk_constraint_metric": round(_safe_float(explain.get("risk_constraint_metric"), 0.0), 6),
            "risk_constraint_satisfied": bool(explain.get("risk_constraint_satisfied", True)),
        },
        "feasible_domain": {
            "blocked_ratio": round(_safe_float(explain.get("blocked_ratio_last"), 0.0), 6),
            "changed_blocked_cells": _safe_int(replan_item.get("changed_blocked_cells"), 0),
        },
        "planner_state": {
            "planner": planner_name,
            "update_mode": str(replan_item.get("update_mode", "rebuild")),
            "runtime_ms": round(runtime_ms, 6),
            "raw_points": _safe_int(replan_item.get("raw_points"), 0),
            "smoothed_points": _safe_int(replan_item.get("smoothed_points"), 0),
            "moved_edges": _safe_int(replan_item.get("moved_edges"), 0),
            "moved_distance_km": round(moved_distance_km, 6),
            "smoothing_feasible": bool(replan_item.get("smoothing_feasible", True)),
            "smoothing_fallback_reason": str(replan_item.get("smoothing_fallback_reason", "")),
        },
        "execution_state": {
            "cumulative_distance_km": round(cumulative_distance_km, 6),
            "cumulative_runtime_ms": round(cumulative_runtime_ms, 6),
            "route_cost_effective_km": round(_safe_float(explain.get("route_cost_effective_km"), 0.0), 6),
            "route_cost_risk_extra_km": round(_safe_float(explain.get("route_cost_risk_extra_km"), 0.0), 6),
            "caution_len_km": round(_safe_float(explain.get("caution_len_km"), 0.0), 6),
        },
    }


def build_dynamic_state_sequence(
    *,
    settings: Settings,
    timestamps: list[str],
    start: dict[str, float],
    goal: dict[str, float],
    policy: dict[str, Any],
    explain: dict[str, Any],
    model_version: str = "unet_v1",
) -> dict[str, Any]:
    now = _utc_now_iso()
    version_snapshot = build_dynamic_state_version_snapshot(settings=settings, model_version=model_version)
    replans = explain.get("dynamic_replans", [])
    if not isinstance(replans, list):
        replans = []

    cumulative_distance_km = 0.0
    cumulative_runtime_ms = 0.0
    steps: list[dict[str, Any]] = []
    for idx, item in enumerate(replans):
        if not isinstance(item, dict):
            continue
        step_ts = str(item.get("timestamp", timestamps[min(idx, max(0, len(timestamps) - 1))] if timestamps else ""))
        step = _step_record(
            step_index=idx,
            timestamp=step_ts,
            replan_item=item,
            explain=explain,
            cumulative_distance_km=cumulative_distance_km,
            cumulative_runtime_ms=cumulative_runtime_ms,
        )
        cumulative_distance_km = float(step["execution_state"]["cumulative_distance_km"])
        cumulative_runtime_ms = float(step["execution_state"]["cumulative_runtime_ms"])
        steps.append(step)

    sequence_id = _sequence_id(timestamps=timestamps, policy=policy, start=start, goal=goal)
    return {
        "sequence_id": sequence_id,
        "created_at": now,
        "updated_at": now,
        "status": "complete",
        "version_snapshot": version_snapshot,
        "timeline": {
            "timestamps": [str(x) for x in timestamps],
            "advance_steps": _safe_int(explain.get("dynamic_advance_steps"), 0),
            "step_count": len(steps),
        },
        "request": {
            "start": {"lat": _safe_float(start.get("lat"), 0.0), "lon": _safe_float(start.get("lon"), 0.0)},
            "goal": {"lat": _safe_float(goal.get("lat"), 0.0), "lon": _safe_float(goal.get("lon"), 0.0)},
            "policy": dict(policy),
        },
        "summary": {
            "planner": str(explain.get("planner", "")),
            "distance_km": round(_safe_float(explain.get("distance_km"), 0.0), 6),
            "distance_nm": round(_safe_float(explain.get("distance_nm"), 0.0), 6),
            "caution_len_km": round(_safe_float(explain.get("caution_len_km"), 0.0), 6),
            "replan_count": len(steps),
            "replan_runtime_ms_total": round(_safe_float(explain.get("replan_runtime_ms_total"), 0.0), 6),
            "replan_runtime_ms_mean": round(_safe_float(explain.get("replan_runtime_ms_mean"), 0.0), 6),
            "dynamic_incremental_steps": _safe_int(explain.get("dynamic_incremental_steps"), 0),
            "dynamic_rebuild_steps": _safe_int(explain.get("dynamic_rebuild_steps"), 0),
        },
        "steps": steps,
    }


def validate_dynamic_state_sequence(payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    required_top = ("sequence_id", "version_snapshot", "timeline", "summary", "steps")
    for key in required_top:
        if key not in payload:
            issues.append(f"missing_top:{key}")
    snapshot = payload.get("version_snapshot", {})
    for key in ("dataset_version", "model_version", "plan_version", "eval_version", "state_version"):
        if not isinstance(snapshot, dict) or not str(snapshot.get(key, "")).strip():
            issues.append(f"missing_version:{key}")
    steps = payload.get("steps", [])
    if not isinstance(steps, list):
        issues.append("steps_not_list")
        return issues
    required_step = ("env_state", "risk_state", "feasible_domain", "planner_state", "execution_state")
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            issues.append(f"step_not_dict:{idx}")
            continue
        for key in required_step:
            if key not in step:
                issues.append(f"missing_step:{idx}:{key}")
    return issues


def save_dynamic_state_sequence(*, settings: Settings, sequence: dict[str, Any]) -> dict[str, str]:
    root = _state_root(settings)
    sequence_id = str(sequence.get("sequence_id") or "")
    if not sequence_id:
        sequence_id = _sequence_id(
            timestamps=list(sequence.get("timeline", {}).get("timestamps", [])),
            policy=dict(sequence.get("request", {}).get("policy", {})),
            start=dict(sequence.get("request", {}).get("start", {})),
            goal=dict(sequence.get("request", {}).get("goal", {})),
        )
        sequence["sequence_id"] = sequence_id
    sequence["updated_at"] = _utc_now_iso()

    sequence_dir = root / sequence_id
    sequence_dir.mkdir(parents=True, exist_ok=True)
    sequence_path = sequence_dir / "sequence.json"
    sequence_path.write_text(json.dumps(sequence, ensure_ascii=False, indent=2), encoding="utf-8")

    steps = sequence.get("steps", [])
    last_step = dict(steps[-1]) if isinstance(steps, list) and steps else {}
    checkpoint = {
        "sequence_id": sequence_id,
        "state_version": str(sequence.get("version_snapshot", {}).get("state_version", settings.dynamic_state_version)),
        "updated_at": _utc_now_iso(),
        "last_step_index": _safe_int(last_step.get("step_index"), -1),
        "next_step_index": _safe_int(last_step.get("step_index"), -1) + 1,
        "last_timestamp": str(last_step.get("timestamp", "")),
        "last_update_mode": str(last_step.get("planner_state", {}).get("update_mode", "")),
    }
    checkpoint_path = sequence_dir / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_ptr = root / "latest_sequence.json"
    latest_ptr.write_text(
        json.dumps(
            {
                "sequence_id": sequence_id,
                "sequence_file": str(sequence_path),
                "checkpoint_file": str(checkpoint_path),
                "updated_at": checkpoint["updated_at"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "sequence_id": sequence_id,
        "sequence_file": str(sequence_path),
        "checkpoint_file": str(checkpoint_path),
    }


def resolve_dynamic_state_sequence_path(*, settings: Settings, sequence_id_or_path: str) -> Path:
    candidate = Path(sequence_id_or_path)
    if candidate.exists():
        if candidate.is_dir():
            p = candidate / "sequence.json"
            if p.exists():
                return p
        return candidate
    root = _state_root(settings)
    as_dir = root / sequence_id_or_path / "sequence.json"
    if as_dir.exists():
        return as_dir
    as_file = root / f"{sequence_id_or_path}.json"
    if as_file.exists():
        return as_file
    raise FileNotFoundError(f"dynamic state sequence not found: {sequence_id_or_path}")


def load_dynamic_state_sequence(*, settings: Settings, sequence_id_or_path: str) -> dict[str, Any]:
    path = resolve_dynamic_state_sequence_path(settings=settings, sequence_id_or_path=sequence_id_or_path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_dynamic_state_checkpoint(*, settings: Settings, sequence_id_or_path: str) -> dict[str, Any]:
    path = resolve_dynamic_state_sequence_path(settings=settings, sequence_id_or_path=sequence_id_or_path)
    checkpoint = path.parent / "checkpoint.json"
    if checkpoint.exists():
        return json.loads(checkpoint.read_text(encoding="utf-8"))
    sequence = load_dynamic_state_sequence(settings=settings, sequence_id_or_path=sequence_id_or_path)
    steps = sequence.get("steps", [])
    last_step = dict(steps[-1]) if isinstance(steps, list) and steps else {}
    return {
        "sequence_id": str(sequence.get("sequence_id", "")),
        "state_version": str(sequence.get("version_snapshot", {}).get("state_version", settings.dynamic_state_version)),
        "last_step_index": _safe_int(last_step.get("step_index"), -1),
        "next_step_index": _safe_int(last_step.get("step_index"), -1) + 1,
        "last_timestamp": str(last_step.get("timestamp", "")),
        "last_update_mode": str(last_step.get("planner_state", {}).get("update_mode", "")),
        "updated_at": str(sequence.get("updated_at", "")),
    }


def build_resume_context(*, settings: Settings, sequence_id_or_path: str) -> dict[str, Any]:
    checkpoint = load_dynamic_state_checkpoint(settings=settings, sequence_id_or_path=sequence_id_or_path)
    return {
        "sequence_id": str(checkpoint.get("sequence_id", "")),
        "state_version": str(checkpoint.get("state_version", settings.dynamic_state_version)),
        "next_step_index": _safe_int(checkpoint.get("next_step_index"), 0),
        "last_step_index": _safe_int(checkpoint.get("last_step_index"), -1),
        "last_timestamp": str(checkpoint.get("last_timestamp", "")),
        "last_update_mode": str(checkpoint.get("last_update_mode", "")),
    }


def replay_dynamic_steps(
    sequence: dict[str, Any],
    *,
    start_step: int = 0,
    end_step: int | None = None,
) -> Iterator[dict[str, Any]]:
    steps = sequence.get("steps", [])
    if not isinstance(steps, list):
        return
    start = max(0, int(start_step))
    stop = len(steps) if end_step is None else min(len(steps), max(start, int(end_step)))
    for idx in range(start, stop):
        step = steps[idx]
        if isinstance(step, dict):
            yield step
