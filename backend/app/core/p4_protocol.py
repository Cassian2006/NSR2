from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.dataset import DatasetService
from app.core.versioning import build_version_snapshot


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sample_indices(total: int, k: int) -> list[int]:
    if total <= 0 or k <= 0:
        return []
    if k >= total:
        return list(range(total))
    if k == 1:
        return [0]
    step = (total - 1) / float(k - 1)
    indices = [int(round(i * step)) for i in range(k)]
    out: list[int] = []
    seen: set[int] = set()
    for idx in indices:
        idx = max(0, min(total - 1, idx))
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def _build_static_cases(
    *,
    timestamps: list[str],
    starts_goals: list[dict[str, Any]],
    case_count: int,
) -> list[dict[str, Any]]:
    if not timestamps or not starts_goals:
        return []
    ts_indices = _sample_indices(len(timestamps), case_count)
    cases: list[dict[str, Any]] = []
    for case_idx, ts_i in enumerate(ts_indices, start=1):
        pair = starts_goals[(case_idx - 1) % len(starts_goals)]
        ts = timestamps[ts_i]
        cases.append(
            {
                "id": f"S{case_idx:03d}",
                "mode": "static",
                "timestamp": ts,
                "start": dict(pair["start"]),
                "goal": dict(pair["goal"]),
                "tags": ["protocol_frozen", "static"],
            }
        )
    return cases


def _build_dynamic_cases(
    *,
    timestamps: list[str],
    starts_goals: list[dict[str, Any]],
    case_count: int,
    window: int,
    advance_steps: int,
) -> list[dict[str, Any]]:
    if len(timestamps) < 2 or not starts_goals:
        return []
    w = max(2, int(window))
    if len(timestamps) < w:
        return []
    max_start = len(timestamps) - w
    start_indices = _sample_indices(max_start + 1, case_count)
    cases: list[dict[str, Any]] = []
    for case_idx, start_i in enumerate(start_indices, start=1):
        pair = starts_goals[(case_idx - 1) % len(starts_goals)]
        win = timestamps[start_i : start_i + w]
        if len(win) < 2:
            continue
        cases.append(
            {
                "id": f"D{case_idx:03d}",
                "mode": "dynamic",
                "timestamps": win,
                "window_start": win[0],
                "window_end": win[-1],
                "advance_steps": int(max(1, advance_steps)),
                "start": dict(pair["start"]),
                "goal": dict(pair["goal"]),
                "tags": ["protocol_frozen", "dynamic"],
            }
        )
    return cases


def compute_protocol_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def build_p4_eval_protocol(
    *,
    settings,
    model_version: str,
    static_case_count: int,
    dynamic_case_count: int,
    dynamic_window: int,
    dynamic_advance_steps: int,
    planners: list[str] | None = None,
) -> dict[str, Any]:
    dataset = DatasetService()
    timestamps = dataset.list_timestamps(month="all")
    months = dataset.list_months()
    version_snapshot = build_version_snapshot(settings=settings, model_version=model_version)
    starts_goals = [
        {"name": "west_to_east", "start": {"lat": 70.5, "lon": 30.0}, "goal": {"lat": 72.0, "lon": 150.0}},
        {"name": "east_to_west", "start": {"lat": 72.0, "lon": 150.0}, "goal": {"lat": 70.5, "lon": 30.0}},
        {"name": "mid_arc", "start": {"lat": 71.0, "lon": 60.0}, "goal": {"lat": 73.0, "lon": 120.0}},
    ]
    static_cases = _build_static_cases(
        timestamps=timestamps,
        starts_goals=starts_goals,
        case_count=max(1, int(static_case_count)),
    )
    dynamic_cases = _build_dynamic_cases(
        timestamps=timestamps,
        starts_goals=starts_goals,
        case_count=max(1, int(dynamic_case_count)),
        window=max(2, int(dynamic_window)),
        advance_steps=max(1, int(dynamic_advance_steps)),
    )
    protocol: dict[str, Any] = {
        "protocol_version": "p4_eval_protocol_v1",
        "frozen_at": _utc_now_iso(),
        "version_snapshot": version_snapshot,
        "scope": {
            "timestamp_count": len(timestamps),
            "timestamp_start": timestamps[0] if timestamps else "",
            "timestamp_end": timestamps[-1] if timestamps else "",
            "months": months,
        },
        "planners": planners or ["astar", "dstar_lite", "any_angle", "hybrid_astar"],
        "policy_matrix": [
            {
                "id": "baseline",
                "caution_mode": "tie_breaker",
                "corridor_bias": 0.2,
                "risk_mode": "balanced",
                "risk_constraint_mode": "none",
            },
            {
                "id": "risk_conservative",
                "caution_mode": "minimize",
                "corridor_bias": 0.15,
                "risk_mode": "conservative",
                "risk_constraint_mode": "chance",
            },
            {
                "id": "distance_first",
                "caution_mode": "tie_breaker",
                "corridor_bias": 0.25,
                "risk_mode": "aggressive",
                "risk_constraint_mode": "none",
            },
        ],
        "scenarios": {
            "static": static_cases,
            "dynamic": dynamic_cases,
        },
        "acceptance": {
            "repeatability": {
                "metrics": ["distance_km", "route_cost_effective_km", "caution_len_km"],
                "abs_tol": 1e-6,
                "rel_tol": 1e-6,
            }
        },
    }
    frozen_core = {
        "protocol_version": protocol["protocol_version"],
        "version_snapshot": protocol["version_snapshot"],
        "scope": protocol["scope"],
        "planners": protocol["planners"],
        "policy_matrix": protocol["policy_matrix"],
        "scenarios": protocol["scenarios"],
        "acceptance": protocol["acceptance"],
    }
    protocol["protocol_hash"] = compute_protocol_hash(frozen_core)
    return protocol


def write_protocol(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def compare_repeat_rows(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    metrics: list[str],
    abs_tol: float,
    rel_tol: float,
) -> dict[str, Any]:
    by_id_a = {str(r.get("case_id", "")): r for r in rows_a}
    by_id_b = {str(r.get("case_id", "")): r for r in rows_b}
    ids = sorted(set(by_id_a.keys()) & set(by_id_b.keys()))
    details: list[dict[str, Any]] = []
    violated = False
    for cid in ids:
        a = by_id_a[cid]
        b = by_id_b[cid]
        item = {"case_id": cid, "metrics": []}
        for m in metrics:
            va = float(a.get(m, 0.0))
            vb = float(b.get(m, 0.0))
            abs_diff = abs(va - vb)
            denom = max(abs(va), abs(vb), 1e-12)
            rel_diff = abs_diff / denom
            ok = abs_diff <= float(abs_tol) or rel_diff <= float(rel_tol)
            if not ok:
                violated = True
            item["metrics"].append(
                {
                    "name": m,
                    "run_a": va,
                    "run_b": vb,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                    "ok": ok,
                }
            )
        details.append(item)
    return {
        "status": "pass" if not violated else "fail",
        "case_count": len(ids),
        "details": details,
        "abs_tol": float(abs_tol),
        "rel_tol": float(rel_tol),
    }
