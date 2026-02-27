from __future__ import annotations

import heapq
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.core.geo import load_grid_geo
from app.core.risk_field import get_risk_layer
from app.core.uncertainty_runtime import (
    build_uncertainty_penalty_map,
    calibrate_uncertainty_grid,
    load_uncertainty_calibration_profile,
)
from app.model.infer import run_unet_inference


EARTH_RADIUS_KM = 6371.0088
DSTAR_INCREMENTAL_CHANGED_RATIO = 0.08
DSTAR_INCREMENTAL_CHANGED_MIN_CELLS = 64
PATH_METRICS_CACHE_MAX = 512


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class PlanningError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlanResult:
    route_geojson: dict
    explain: dict


@dataclass(frozen=True)
class GridBounds:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass(frozen=True)
class GridState:
    timestamp: str
    geo: object
    bounds: GridBounds
    blocked: np.ndarray
    caution: np.ndarray
    ais_norm: np.ndarray
    free_cells: np.ndarray
    near_blocked: np.ndarray | None = None
    uncertainty_calibrated: np.ndarray | None = None
    uncertainty_penalty: np.ndarray | None = None
    uncertainty_meta: dict | None = None
    risk_penalty: np.ndarray | None = None
    risk_meta: dict | None = None
    risk_penalty_by_mode: dict[str, np.ndarray] | None = None
    risk_meta_by_mode: dict[str, dict] | None = None


def _neighbors(r: int, c: int, h: int, w: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


def _optional_array_diff_mask(
    prev: np.ndarray | None,
    curr: np.ndarray | None,
    shape: tuple[int, int],
    *,
    atol: float = 1e-6,
) -> np.ndarray:
    if prev is None and curr is None:
        return np.zeros(shape, dtype=bool)
    if prev is None or curr is None:
        return np.ones(shape, dtype=bool)
    return np.abs(prev - curr) > atol


def _count_changed_directed_edges(changed_cells: np.ndarray, *, h: int, w: int) -> int:
    if changed_cells.size == 0:
        return 0
    directed: set[tuple[int, int, int, int]] = set()
    for rc in changed_cells:
        r = int(rc[0])
        c = int(rc[1])
        for nr, nc in _neighbors(r, c, h, w):
            directed.add((r, c, nr, nc))
            directed.add((nr, nc, r, c))
    return len(directed)


def _grid_directed_edge_count(h: int, w: int) -> int:
    total = 0
    for r in range(h):
        for c in range(w):
            total += sum(1 for _ in _neighbors(r, c, h, w))
    return int(total)


def _metric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"total": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "total": float(arr.sum()),
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def _memory_usage_mb() -> float:
    """Best-effort process RSS in MB, cross-platform fallback."""
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if os.name == "posix" and os.uname().sysname.lower() == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        return 0.0


def _heuristic_km(
    r: int,
    c: int,
    goal_r: int,
    goal_c: int,
    *,
    km_per_row: float,
    km_per_col_min: float,
) -> float:
    dr = abs(goal_r - r)
    dc = abs(goal_c - c)
    return math.hypot(dr * km_per_row, dc * km_per_col_min)


def _reconstruct_path(came_from: dict[tuple[int, int], tuple[int, int]], cur: tuple[int, int]) -> list[tuple[int, int]]:
    path = [cur]
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def _nearest_unblocked(
    cell: tuple[int, int],
    blocked: np.ndarray,
    free_cells: np.ndarray,
) -> tuple[int, int]:
    r0, c0 = cell
    if not blocked[r0, c0]:
        return cell
    if free_cells.size == 0:
        raise PlanningError("No navigable cells available in current grid.")
    dr = free_cells[:, 0] - r0
    dc = free_cells[:, 1] - c0
    idx = int(np.argmin(dr * dr + dc * dc))
    rr, cc = free_cells[idx]
    return int(rr), int(cc)


def _line_of_sight(a: tuple[int, int], b: tuple[int, int], blocked: np.ndarray) -> bool:
    """Supercover LOS check; false when any touched blocked cell is crossed."""
    for rr, cc in _trace_line_cells(a, b, blocked.shape):
        if blocked[rr, cc]:
            return False
    return True


def _smooth_cells_los(cells: list[tuple[int, int]], blocked: np.ndarray) -> list[tuple[int, int]]:
    """Greedy farthest-visible simplification, preserving navigability."""
    if len(cells) <= 2:
        return cells
    out = [cells[0]]
    i = 0
    n = len(cells)
    while i < n - 1:
        j = n - 1
        while j > i + 1:
            if _line_of_sight(cells[i], cells[j], blocked):
                break
            j -= 1
        out.append(cells[j])
        i = j
    return out


def _turn_angle_deg(
    prev_rc: tuple[int, int],
    cur_rc: tuple[int, int],
    next_rc: tuple[int, int],
) -> float:
    v1 = np.asarray([cur_rc[0] - prev_rc[0], cur_rc[1] - prev_rc[1]], dtype=np.float64)
    v2 = np.asarray([next_rc[0] - cur_rc[0], next_rc[1] - cur_rc[1]], dtype=np.float64)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-12 or n2 <= 1e-12:
        return 0.0
    cosv = float(np.dot(v1, v2) / (n1 * n2))
    cosv = float(np.clip(cosv, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))


def _max_turn_angle_deg(cells: list[tuple[int, int]]) -> float:
    if len(cells) < 3:
        return 0.0
    return max(_turn_angle_deg(cells[i - 1], cells[i], cells[i + 1]) for i in range(1, len(cells) - 1))


def _is_path_feasible(cells: list[tuple[int, int]], blocked: np.ndarray) -> bool:
    if len(cells) < 2:
        return True
    for i in range(1, len(cells)):
        if not _line_of_sight(cells[i - 1], cells[i], blocked):
            return False
    return True


def _mean_penalty_on_cells(penalty: np.ndarray | None, cells: list[tuple[int, int]]) -> float:
    if penalty is None or not cells:
        return 0.0
    vals = [float(penalty[r, c]) for r, c in cells]
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _cells_blocked_ratio(cells: list[tuple[int, int]], blocked: np.ndarray) -> float:
    if not cells:
        return 0.0
    hits = 0
    for r, c in cells:
        if blocked[r, c]:
            hits += 1
    return float(hits) / float(max(1, len(cells)))


def _align_cached_path(cells: list[tuple[int, int]], current_start: tuple[int, int]) -> list[tuple[int, int]]:
    if not cells:
        return []
    if cells[0] == current_start:
        return cells
    try:
        idx = cells.index(current_start)
        return cells[idx:]
    except ValueError:
        return []


def _smooth_cells_los_constrained(
    cells: list[tuple[int, int]],
    blocked: np.ndarray,
    *,
    max_turn_deg: float,
) -> list[tuple[int, int]]:
    """LOS smoothing with turn-angle bound for more realistic marine trajectories."""
    if len(cells) <= 2:
        return cells
    out = [cells[0]]
    i = 0
    n = len(cells)
    while i < n - 1:
        chosen = i + 1
        for j in range(n - 1, i, -1):
            if not _line_of_sight(cells[i], cells[j], blocked):
                continue
            if len(out) >= 2:
                turn_deg = _turn_angle_deg(out[-2], out[-1], cells[j])
                if turn_deg > float(max_turn_deg):
                    continue
            chosen = j
            break
        out.append(cells[chosen])
        i = chosen
    return out


def _cells_to_coords(geo, cells: list[tuple[int, int]]) -> list[list[float]]:
    coords: list[list[float]] = []
    for r, c in cells:
        lat, lon = geo.rc_to_latlon(r, c)
        coords.append([lon, lat])
    return coords


def _trace_line_cells(a: tuple[int, int], b: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    """Supercover trace of all touched grid cells from a to b (inclusive endpoints)."""
    r0, c0 = a
    r1, c1 = b
    h, w = shape
    x0, y0 = c0, r0
    x1, y1 = c1, r1
    dx = x1 - x0
    dy = y1 - y0
    nx = abs(dx)
    ny = abs(dy)
    sign_x = 1 if dx > 0 else -1
    sign_y = 1 if dy > 0 else -1

    x, y = x0, y0
    out: list[tuple[int, int]] = []

    def _append(rr: int, cc: int) -> None:
        if 0 <= rr < h and 0 <= cc < w:
            cell = (rr, cc)
            if not out or out[-1] != cell:
                out.append(cell)

    _append(y, x)

    ix = 0
    iy = 0
    while ix < nx or iy < ny:
        decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
        if decision == 0:
            _append(y + sign_y, x)
            _append(y, x + sign_x)
            x += sign_x
            y += sign_y
            ix += 1
            iy += 1
        elif decision < 0:
            x += sign_x
            ix += 1
        else:
            y += sign_y
            iy += 1
        _append(y, x)
    return out


def _build_near_blocked_mask(blocked: np.ndarray) -> np.ndarray:
    near = np.zeros_like(blocked, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(blocked, shift=(dr, dc), axis=(0, 1))
            if dr > 0:
                shifted[:dr, :] = False
            elif dr < 0:
                shifted[dr:, :] = False
            if dc > 0:
                shifted[:, :dc] = False
            elif dc < 0:
                shifted[:, dc:] = False
            near |= shifted
    near &= ~blocked
    return near


def _load_ais_heatmap(settings: Settings, timestamp: str, shape: tuple[int, int]) -> np.ndarray:
    root = settings.ais_heatmap_root
    if root.exists():
        hits = list(root.rglob(f"{timestamp}.npy"))
        if hits:
            arr = np.load(hits[0]).astype(np.float32)
            if arr.shape == shape:
                return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.zeros(shape, dtype=np.float32)


def _load_uncertainty_penalty(
    *,
    settings: Settings,
    timestamp: str,
    model_version: str,
    shape: tuple[int, int],
    enabled: bool,
    uplift_scale_factor: float,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    if not enabled:
        return None, None, {"enabled": False, "applied": False}

    pred_path = settings.pred_root / model_version / f"{timestamp}.npy"
    unc_path = settings.pred_root / model_version / f"{timestamp}_uncertainty.npy"
    if not unc_path.exists():
        try:
            run_unet_inference(
                settings=settings,
                timestamp=timestamp,
                model_version=model_version,
                output_path=pred_path,
            )
        except Exception:
            pass

    if not unc_path.exists():
        return None, None, {
            "enabled": True,
            "applied": False,
            "reason": "uncertainty_file_missing",
        }
    try:
        unc_raw = np.load(unc_path).astype(np.float32)
    except Exception:
        return None, None, {
            "enabled": True,
            "applied": False,
            "reason": "uncertainty_load_failed",
            "uncertainty_path": str(unc_path),
        }
    if unc_raw.shape != shape:
        return None, None, {
            "enabled": True,
            "applied": False,
            "reason": "uncertainty_shape_mismatch",
            "uncertainty_path": str(unc_path),
            "shape": list(unc_raw.shape),
            "expected_shape": [int(shape[0]), int(shape[1])],
        }

    profile = load_uncertainty_calibration_profile(settings=settings, model_version=model_version)
    unc_cal = calibrate_uncertainty_grid(unc_raw, temperature=profile.temperature)
    final_scale = max(0.0, float(profile.uplift_scale) * max(0.0, float(uplift_scale_factor)))
    penalty = build_uncertainty_penalty_map(
        unc_cal,
        threshold=profile.uncertainty_threshold,
        uplift_scale=final_scale,
    )
    meta = {
        "enabled": True,
        "applied": True,
        "uncertainty_path": str(unc_path),
        "profile_available": bool(profile.available),
        "temperature": float(profile.temperature),
        "threshold": float(profile.uncertainty_threshold),
        "uplift_scale": float(final_scale),
        "profile_uplift_scale": float(profile.uplift_scale),
        "profile_source_path": profile.source_path,
        "ece_before": profile.ece_before,
        "ece_after": profile.ece_after,
        "brier_before": profile.brier_before,
        "brier_after": profile.brier_after,
        "uncertainty_mean": float(np.nanmean(unc_cal)),
        "uncertainty_p90": float(np.nanpercentile(unc_cal, 90)),
        "penalty_mean": float(np.nanmean(penalty)),
        "penalty_p90": float(np.nanpercentile(penalty, 90)),
        "high_uncertainty_ratio": float(np.mean(unc_cal > profile.uncertainty_threshold)),
    }
    return unc_cal, penalty, meta


def _risk_mode_profile(risk_mode: str) -> tuple[str, float]:
    mode = risk_mode.strip().lower()
    if mode == "conservative":
        return "risk_p90", 0.55
    if mode == "aggressive":
        return "risk_mean", 0.12
    return "risk_mean", 0.30


def _risk_raw_values_from_penalty_values(penalty_vals: list[float], risk_lambda: float) -> list[float]:
    lam = float(max(0.0, risk_lambda))
    if lam <= 1e-8:
        return [0.0 for _ in penalty_vals]
    return [float(np.clip(v / lam, 0.0, 1.0)) for v in penalty_vals]


def _evaluate_risk_constraint(
    *,
    mode: str,
    budget: float,
    confidence_level: float,
    raw_risk_values: list[float],
) -> dict:
    mode_key = str(mode).strip().lower()
    budget_safe = float(max(0.0, budget))
    conf = float(np.clip(confidence_level, 0.5, 0.999))
    values = [float(np.clip(v, 0.0, 1.0)) for v in raw_risk_values]
    n = int(len(values))

    if mode_key == "none" or n == 0:
        return {
            "mode": mode_key,
            "metric_name": "none",
            "metric": 0.0,
            "budget": budget_safe,
            "usage": 0.0,
            "satisfied": True,
            "sample_count": n,
            "confidence_level": conf,
            "tail_count": 0,
        }

    if mode_key == "chance":
        metric = float(np.mean(np.asarray(values) > conf))
        metric_name = "chance_violation_ratio"
        tail_count = int(np.count_nonzero(np.asarray(values) > conf))
    elif mode_key == "cvar":
        tail_frac = max(1e-6, 1.0 - conf)
        tail_n = max(1, int(math.ceil(n * tail_frac)))
        sorted_vals = sorted(values)
        tail = sorted_vals[-tail_n:]
        metric = float(np.mean(tail))
        metric_name = "cvar_tail_mean"
        tail_count = int(tail_n)
    else:
        raise PlanningError(f"Unsupported risk_constraint_mode={mode}, expected one of ['none','chance','cvar']")

    budget_denom = max(1e-6, budget_safe)
    usage = float(metric / budget_denom)
    satisfied = bool(metric <= budget_safe + 1e-12)
    return {
        "mode": mode_key,
        "metric_name": metric_name,
        "metric": float(metric),
        "budget": budget_safe,
        "usage": usage,
        "satisfied": satisfied,
        "sample_count": n,
        "confidence_level": conf,
        "tail_count": int(tail_count),
    }


def _load_risk_penalty(
    *,
    settings: Settings,
    timestamp: str,
    model_version: str,
    shape: tuple[int, int],
    risk_mode: str,
    risk_weight_scale: float,
) -> tuple[np.ndarray | None, dict]:
    layer_name, mode_lambda = _risk_mode_profile(risk_mode)
    weight = max(0.0, float(mode_lambda) * max(0.0, float(risk_weight_scale)))
    if weight <= 1e-8:
        return None, {
            "risk_mode": risk_mode,
            "risk_layer": layer_name,
            "risk_lambda": 0.0,
            "applied": False,
            "reason": "zero_weight",
        }
    try:
        risk_layer = get_risk_layer(
            settings=settings,
            timestamp=timestamp,
            layer=layer_name,
            model_version=model_version,
            force_refresh=False,
        )
    except Exception as exc:
        return None, {
            "risk_mode": risk_mode,
            "risk_layer": layer_name,
            "risk_lambda": float(weight),
            "applied": False,
            "reason": f"risk_layer_error:{exc}",
        }
    if risk_layer is None:
        return None, {
            "risk_mode": risk_mode,
            "risk_layer": layer_name,
            "risk_lambda": float(weight),
            "applied": False,
            "reason": "risk_layer_missing",
        }
    if risk_layer.shape != shape:
        return None, {
            "risk_mode": risk_mode,
            "risk_layer": layer_name,
            "risk_lambda": float(weight),
            "applied": False,
            "reason": "risk_layer_shape_mismatch",
            "shape": list(risk_layer.shape),
            "expected_shape": [int(shape[0]), int(shape[1])],
        }
    risk_norm = np.clip(np.nan_to_num(risk_layer.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    penalty = (risk_norm * float(weight)).astype(np.float32)
    return penalty, {
        "risk_mode": risk_mode,
        "risk_layer": layer_name,
        "risk_lambda": float(weight),
        "applied": True,
        "risk_mean": float(np.nanmean(risk_norm)),
        "risk_p90": float(np.nanpercentile(risk_norm, 90)),
        "penalty_mean": float(np.nanmean(penalty)),
        "penalty_p90": float(np.nanpercentile(penalty, 90)),
    }


def _state_risk_penalty_for_mode(state: GridState, mode: str) -> np.ndarray | None:
    mode_key = str(mode).strip().lower()
    if isinstance(state.risk_penalty_by_mode, dict) and mode_key in state.risk_penalty_by_mode:
        return state.risk_penalty_by_mode[mode_key]
    return state.risk_penalty


def _state_risk_meta_for_mode(state: GridState, mode: str) -> dict:
    mode_key = str(mode).strip().lower()
    if isinstance(state.risk_meta_by_mode, dict) and mode_key in state.risk_meta_by_mode:
        return dict(state.risk_meta_by_mode[mode_key])
    return dict(state.risk_meta or {"risk_mode": mode_key, "risk_lambda": 0.0, "applied": False})


def _dynamic_risk_stage_and_target_mode(
    *,
    enabled: bool,
    cumulative_risk_extra_km: float,
    budget_km: float,
    base_mode: str,
    warn_mode: str,
    hard_mode: str,
    warn_ratio: float,
    hard_ratio: float,
) -> tuple[str, str, float]:
    mode_base = str(base_mode).strip().lower()
    if not enabled:
        return "disabled", mode_base, 0.0
    budget = float(max(0.0, budget_km))
    usage = float(cumulative_risk_extra_km / budget) if budget > 1e-8 else 0.0
    usage = float(max(0.0, usage))
    if usage >= float(hard_ratio):
        return "hard", str(hard_mode).strip().lower(), usage
    if usage >= float(warn_ratio):
        return "warn", str(warn_mode).strip().lower(), usage
    return "normal", mode_base, usage


def _policy_scalars(caution_mode: str, corridor_bias: float) -> tuple[float, float, float]:
    if caution_mode == "tie_breaker":
        caution_penalty = 0.22
        corridor_reward_scale = 0.10
        near_blocked_penalty = 0.06
    elif caution_mode == "budget":
        caution_penalty = 0.35
        corridor_reward_scale = 0.06
        near_blocked_penalty = 0.08
    elif caution_mode == "minimize":
        caution_penalty = 0.55
        corridor_reward_scale = 0.03
        near_blocked_penalty = 0.10
    else:  # strict
        caution_penalty = 0.0
        corridor_reward_scale = 0.0
        near_blocked_penalty = 0.0
    corridor_reward = max(0.0, float(corridor_bias)) * corridor_reward_scale
    return caution_penalty, corridor_reward, near_blocked_penalty


def _grid_resolution_km(bounds: GridBounds, h: int, w: int) -> tuple[float, float]:
    lat_step = (bounds.lat_max - bounds.lat_min) / max(1, h - 1)
    lon_step = (bounds.lon_max - bounds.lon_min) / max(1, w - 1)
    km_per_row = 111.32 * lat_step
    km_per_col_min = 111.32 * max(0.05, math.cos(math.radians(bounds.lat_max))) * lon_step
    return km_per_row, km_per_col_min


def _load_grid_state(
    *,
    settings: Settings,
    timestamp: str,
    model_version: str,
    blocked_sources: list[str],
    caution_mode: str,
    uncertainty_uplift: bool = True,
    uncertainty_uplift_scale: float = 1.0,
    risk_mode: str = "balanced",
    risk_weight_scale: float = 1.0,
    risk_modes_to_load: list[str] | None = None,
) -> GridState:
    ann_dir = settings.annotation_pack_root / timestamp
    blocked_path = ann_dir / "blocked_mask.npy"
    if not blocked_path.exists():
        raise PlanningError(f"blocked_mask missing for timestamp={timestamp}: {blocked_path}")
    blocked_bathy = np.load(blocked_path).astype(np.uint8) > 0
    h, w = blocked_bathy.shape
    geo = load_grid_geo(settings, timestamp=timestamp, shape=(h, w))

    pred_path = settings.pred_root / model_version / f"{timestamp}.npy"
    if ("unet_blocked" in blocked_sources or caution_mode == "tie_breaker" or caution_mode == "strict") and not pred_path.exists():
        run_unet_inference(
            settings=settings,
            timestamp=timestamp,
            model_version=model_version,
            output_path=pred_path,
        )

    unet_pred = np.load(pred_path).astype(np.uint8) if pred_path.exists() else np.zeros((h, w), dtype=np.uint8)
    if unet_pred.shape != (h, w):
        raise PlanningError(f"Pred shape mismatch for timestamp={timestamp}: {unet_pred.shape} vs {(h, w)}")

    blocked = blocked_bathy.copy()
    if "unet_blocked" in blocked_sources:
        blocked |= unet_pred == 2
    if "unet_caution" in blocked_sources or caution_mode == "strict":
        blocked |= unet_pred == 1

    free_cells = np.argwhere(~blocked)
    if free_cells.size == 0:
        raise PlanningError(f"No navigable cells available for timestamp={timestamp}")

    caution = unet_pred == 1
    ais = _load_ais_heatmap(settings, timestamp=timestamp, shape=(h, w))
    ais = np.clip(ais, 0.0, None)
    ais_max = float(ais.max())
    ais_norm = ais / ais_max if ais_max > 1e-8 else np.zeros_like(ais)
    unc_cal, unc_penalty, unc_meta = _load_uncertainty_penalty(
        settings=settings,
        timestamp=timestamp,
        model_version=model_version,
        shape=(h, w),
        enabled=uncertainty_uplift,
        uplift_scale_factor=uncertainty_uplift_scale,
    )
    risk_penalty, risk_meta = _load_risk_penalty(
        settings=settings,
        timestamp=timestamp,
        model_version=model_version,
        shape=(h, w),
        risk_mode=risk_mode,
        risk_weight_scale=risk_weight_scale,
    )
    risk_penalty_by_mode: dict[str, np.ndarray] | None = None
    risk_meta_by_mode: dict[str, dict] | None = None
    if risk_modes_to_load:
        risk_penalty_by_mode = {}
        risk_meta_by_mode = {}
        for mode_raw in risk_modes_to_load:
            mode_key = str(mode_raw).strip().lower()
            if not mode_key:
                continue
            if mode_key == str(risk_mode).strip().lower():
                risk_penalty_by_mode[mode_key] = risk_penalty
                risk_meta_by_mode[mode_key] = dict(risk_meta)
                continue
            pen_mode, meta_mode = _load_risk_penalty(
                settings=settings,
                timestamp=timestamp,
                model_version=model_version,
                shape=(h, w),
                risk_mode=mode_key,
                risk_weight_scale=risk_weight_scale,
            )
            risk_penalty_by_mode[mode_key] = pen_mode
            risk_meta_by_mode[mode_key] = dict(meta_mode)

    bounds = GridBounds(
        lat_min=float(geo.bounds.lat_min),
        lat_max=float(geo.bounds.lat_max),
        lon_min=float(geo.bounds.lon_min),
        lon_max=float(geo.bounds.lon_max),
    )
    return GridState(
        timestamp=timestamp,
        geo=geo,
        bounds=bounds,
        blocked=blocked,
        caution=caution,
        ais_norm=ais_norm,
        free_cells=free_cells,
        near_blocked=_build_near_blocked_mask(blocked),
        uncertainty_calibrated=unc_cal,
        uncertainty_penalty=unc_penalty,
        uncertainty_meta=unc_meta,
        risk_penalty=risk_penalty,
        risk_meta=risk_meta,
        risk_penalty_by_mode=risk_penalty_by_mode,
        risk_meta_by_mode=risk_meta_by_mode,
    )


def _transition_cost(
    *,
    from_rc: tuple[int, int],
    to_rc: tuple[int, int],
    geo,
    caution: np.ndarray,
    ais_norm: np.ndarray,
    caution_penalty: float,
    corridor_reward: float,
    near_blocked: np.ndarray | None = None,
    near_blocked_penalty: float = 0.0,
    uncertainty_penalty: np.ndarray | None = None,
    risk_penalty: np.ndarray | None = None,
) -> float:
    fr, fc = from_rc
    tr, tc = to_rc
    lat0, lon0 = geo.rc_to_latlon(fr, fc)
    lat1, lon1 = geo.rc_to_latlon(tr, tc)
    step_km = haversine_km(lat0, lon0, lat1, lon1)
    mult = 1.0
    if caution[tr, tc]:
        mult += caution_penalty
    elif near_blocked is not None and near_blocked[tr, tc]:
        mult += near_blocked_penalty
    if uncertainty_penalty is not None:
        mult += float(uncertainty_penalty[tr, tc])
    if risk_penalty is not None:
        mult += float(risk_penalty[tr, tc])
    mult -= corridor_reward * float(ais_norm[tr, tc])
    mult = max(0.15, mult)
    return step_km * mult


def _segment_cell_stats(
    *,
    from_rc: tuple[int, int],
    to_rc: tuple[int, int],
    caution: np.ndarray,
    ais_norm: np.ndarray,
    near_blocked: np.ndarray | None = None,
    uncertainty_penalty: np.ndarray | None = None,
    risk_penalty: np.ndarray | None = None,
) -> tuple[float, float, float, float, float]:
    """Return (caution_ratio, corridor_mean, near_blocked_ratio, uncertainty_penalty_mean, risk_penalty_mean)."""
    h, w = caution.shape
    traced = _trace_line_cells(from_rc, to_rc, (h, w))
    sampled = traced[1:] if len(traced) > 1 else traced
    if not sampled:
        return 0.0, 0.0, 0.0

    caution_hits = 0
    near_hits = 0
    corridor_vals: list[float] = []
    uncertainty_vals: list[float] = []
    risk_vals: list[float] = []
    for rr, rc in sampled:
        caution_hits += int(bool(caution[rr, rc]))
        if near_blocked is not None:
            near_hits += int(bool(near_blocked[rr, rc]))
        corridor_vals.append(float(ais_norm[rr, rc]))
        if uncertainty_penalty is not None:
            uncertainty_vals.append(float(uncertainty_penalty[rr, rc]))
        if risk_penalty is not None:
            risk_vals.append(float(risk_penalty[rr, rc]))

    n = len(sampled)
    caution_ratio = float(caution_hits / max(1, n))
    near_ratio = float(near_hits / max(1, n))
    corridor_mean = float(np.mean(corridor_vals) if corridor_vals else 0.0)
    uncertainty_mean = float(np.mean(uncertainty_vals) if uncertainty_vals else 0.0)
    risk_mean = float(np.mean(risk_vals) if risk_vals else 0.0)
    return caution_ratio, corridor_mean, near_ratio, uncertainty_mean, risk_mean


def _transition_cost_segment(
    *,
    from_rc: tuple[int, int],
    to_rc: tuple[int, int],
    geo,
    caution: np.ndarray,
    ais_norm: np.ndarray,
    caution_penalty: float,
    corridor_reward: float,
    near_blocked: np.ndarray | None = None,
    near_blocked_penalty: float = 0.0,
    uncertainty_penalty: np.ndarray | None = None,
    risk_penalty: np.ndarray | None = None,
) -> float:
    fr, fc = from_rc
    tr, tc = to_rc
    lat0, lon0 = geo.rc_to_latlon(fr, fc)
    lat1, lon1 = geo.rc_to_latlon(tr, tc)
    step_km = haversine_km(lat0, lon0, lat1, lon1)
    caution_ratio, corridor_mean, near_ratio, uncertainty_mean, risk_mean = _segment_cell_stats(
        from_rc=from_rc,
        to_rc=to_rc,
        caution=caution,
        ais_norm=ais_norm,
        near_blocked=near_blocked,
        uncertainty_penalty=uncertainty_penalty,
        risk_penalty=risk_penalty,
    )
    mult = 1.0 + caution_penalty * caution_ratio + near_blocked_penalty * near_ratio + uncertainty_mean + risk_mean
    mult -= corridor_reward * corridor_mean
    mult = max(0.15, mult)
    return step_km * mult


def _collect_path_metrics(
    *,
    cells: list[tuple[int, int]],
    geo,
    caution: np.ndarray,
    ais_norm: np.ndarray,
    near_blocked: np.ndarray,
    caution_penalty: float,
    corridor_reward: float,
    uncertainty_penalty: np.ndarray | None = None,
    risk_penalty: np.ndarray | None = None,
) -> dict:
    if len(cells) < 2:
        return {
            "distance_km": 0.0,
            "base_distance_km": 0.0,
            "caution_len_km": 0.0,
            "cost_caution_extra_km": 0.0,
            "cost_corridor_discount_km": 0.0,
            "cost_uncertainty_extra_km": 0.0,
            "cost_risk_extra_km": 0.0,
            "caution_hits": 0,
            "near_blocked_hits": 0,
            "sample_count": 0,
            "corridor_vals": [],
            "uncertainty_penalty_vals": [],
            "risk_penalty_vals": [],
        }

    h, w = caution.shape
    total_distance_km = 0.0
    caution_len_km = 0.0
    cost_caution_extra_km = 0.0
    cost_corridor_discount_km = 0.0
    cost_uncertainty_extra_km = 0.0
    cost_risk_extra_km = 0.0
    caution_hits = 0
    near_blocked_hits = 0
    sample_count = 0
    corridor_vals: list[float] = []
    uncertainty_penalty_vals: list[float] = []
    risk_penalty_vals: list[float] = []

    for idx in range(1, len(cells)):
        pr, pc = cells[idx - 1]
        cr, cc = cells[idx]
        plat, plon = geo.rc_to_latlon(pr, pc)
        clat, clon = geo.rc_to_latlon(cr, cc)
        segment_km = haversine_km(plat, plon, clat, clon)
        total_distance_km += segment_km

        traced = _trace_line_cells((pr, pc), (cr, cc), (h, w))
        sampled = traced[1:] if len(traced) > 1 else traced
        if not sampled:
            continue

        caution_seg_hits = 0
        near_blocked_seg_hits = 0
        corridor_seg_vals: list[float] = []
        uncertainty_seg_vals: list[float] = []
        risk_seg_vals: list[float] = []
        for rr, rc in sampled:
            caution_seg_hits += int(bool(caution[rr, rc]))
            near_blocked_seg_hits += int(bool(near_blocked[rr, rc]))
            corridor_seg_vals.append(float(ais_norm[rr, rc]))
            if uncertainty_penalty is not None:
                uncertainty_seg_vals.append(float(uncertainty_penalty[rr, rc]))
            if risk_penalty is not None:
                risk_seg_vals.append(float(risk_penalty[rr, rc]))

        seg_samples = len(sampled)
        sample_count += seg_samples
        caution_hits += caution_seg_hits
        near_blocked_hits += near_blocked_seg_hits
        corridor_vals.extend(corridor_seg_vals)

        caution_ratio = float(caution_seg_hits / max(1, seg_samples))
        corridor_mean = float(np.mean(corridor_seg_vals) if corridor_seg_vals else 0.0)
        uncertainty_mean = float(np.mean(uncertainty_seg_vals) if uncertainty_seg_vals else 0.0)
        risk_mean = float(np.mean(risk_seg_vals) if risk_seg_vals else 0.0)

        caution_len_km += segment_km * caution_ratio
        cost_caution_extra_km += segment_km * caution_penalty * caution_ratio
        cost_corridor_discount_km += segment_km * corridor_reward * corridor_mean
        cost_uncertainty_extra_km += segment_km * uncertainty_mean
        cost_risk_extra_km += segment_km * risk_mean
        uncertainty_penalty_vals.extend(uncertainty_seg_vals)
        risk_penalty_vals.extend(risk_seg_vals)

    return {
        "distance_km": total_distance_km,
        "base_distance_km": total_distance_km,
        "caution_len_km": caution_len_km,
        "cost_caution_extra_km": cost_caution_extra_km,
        "cost_corridor_discount_km": cost_corridor_discount_km,
        "cost_uncertainty_extra_km": cost_uncertainty_extra_km,
        "cost_risk_extra_km": cost_risk_extra_km,
        "caution_hits": caution_hits,
        "near_blocked_hits": near_blocked_hits,
        "sample_count": sample_count,
        "corridor_vals": corridor_vals,
        "uncertainty_penalty_vals": uncertainty_penalty_vals,
        "risk_penalty_vals": risk_penalty_vals,
    }


def _build_display_coordinates(coords: list[list[float]], iterations: int = 2) -> list[list[float]]:
    """Chaikin smoothing for display only; does not affect planning metrics."""
    if len(coords) < 3:
        return coords
    points = np.asarray(coords, dtype=np.float64)
    for _ in range(max(1, iterations)):
        new_points = [points[0]]
        for idx in range(len(points) - 1):
            p = points[idx]
            q = points[idx + 1]
            new_points.append(0.75 * p + 0.25 * q)
            new_points.append(0.25 * p + 0.75 * q)
        new_points.append(points[-1])
        points = np.vstack(new_points)
    return [[float(p[0]), float(p[1])] for p in points]


def _expand_cells_supercover(cells: list[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    """Expand sparse path vertices into adjacent supercover cells for rendering safety."""
    if len(cells) < 2:
        return cells
    out: list[tuple[int, int]] = [cells[0]]
    for idx in range(1, len(cells)):
        seg = _trace_line_cells(cells[idx - 1], cells[idx], shape)
        for rc in seg[1:]:
            if not out or out[-1] != rc:
                out.append(rc)
    return out


def _adjacent_blocked_ratio(cells: list[tuple[int, int]], blocked: np.ndarray) -> float:
    if not cells:
        return 0.0
    h, w = blocked.shape
    near = 0
    for r, c in cells:
        is_near = False
        for rr, cc in _neighbors(r, c, h, w):
            if blocked[rr, cc]:
                is_near = True
                break
        if is_near:
            near += 1
    return float(near / max(1, len(cells)))


def _caution_cell_ratio(cells: list[tuple[int, int]], caution: np.ndarray) -> float:
    if not cells:
        return 0.0
    hits = sum(1 for r, c in cells if caution[r, c])
    return float(hits / max(1, len(cells)))


def _run_astar(
    *,
    geo,
    blocked: np.ndarray,
    caution: np.ndarray,
    ais_norm: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    km_per_row: float,
    km_per_col_min: float,
    caution_penalty: float,
    corridor_reward: float,
    near_blocked: np.ndarray | None = None,
    near_blocked_penalty: float = 0.0,
    uncertainty_penalty: np.ndarray | None = None,
    risk_penalty: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    gscore = np.full((h, w), np.inf, dtype=np.float64)
    closed = np.zeros((h, w), dtype=bool)
    came_from: dict[tuple[int, int], tuple[int, int]] = {}

    sr, sc = start
    gr, gc = goal
    gscore[sr, sc] = 0.0
    heap: list[tuple[float, float, int, int]] = []
    heapq.heappush(
        heap,
        (
            _heuristic_km(sr, sc, gr, gc, km_per_row=km_per_row, km_per_col_min=km_per_col_min),
            0.0,
            sr,
            sc,
        ),
    )

    while heap:
        _, cur_g, r, c = heapq.heappop(heap)
        if closed[r, c]:
            continue
        closed[r, c] = True
        if (r, c) == (gr, gc):
            break

        for rr, cc in _neighbors(r, c, h, w):
            if closed[rr, cc] or blocked[rr, cc]:
                continue
            cand = cur_g + _transition_cost(
                from_rc=(r, c),
                to_rc=(rr, cc),
                geo=geo,
                caution=caution,
                ais_norm=ais_norm,
                caution_penalty=caution_penalty,
                corridor_reward=corridor_reward,
                near_blocked=near_blocked,
                near_blocked_penalty=near_blocked_penalty,
                uncertainty_penalty=uncertainty_penalty,
                risk_penalty=risk_penalty,
            )
            if cand < gscore[rr, cc]:
                gscore[rr, cc] = cand
                came_from[(rr, cc)] = (r, c)
                hval = _heuristic_km(rr, cc, gr, gc, km_per_row=km_per_row, km_per_col_min=km_per_col_min)
                heapq.heappush(heap, (cand + hval, cand, rr, cc))

    if not np.isfinite(gscore[gr, gc]):
        raise PlanningError("No feasible route found under current blocked constraints.")
    return _reconstruct_path(came_from, (gr, gc))


def _run_theta_star(
    *,
    geo,
    blocked: np.ndarray,
    caution: np.ndarray,
    ais_norm: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    km_per_row: float,
    km_per_col_min: float,
    caution_penalty: float,
    corridor_reward: float,
    near_blocked: np.ndarray | None = None,
    near_blocked_penalty: float = 0.0,
    uncertainty_penalty: np.ndarray | None = None,
    risk_penalty: np.ndarray | None = None,
    turn_weight: float = 0.03,
) -> list[tuple[int, int]]:
    """Any-angle planning using Theta* with LOS parent rewiring."""
    h, w = blocked.shape
    gscore = np.full((h, w), np.inf, dtype=np.float64)
    closed = np.zeros((h, w), dtype=bool)
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    sr, sc = start
    gr, gc = goal
    gscore[sr, sc] = 0.0
    parent[(sr, sc)] = (sr, sc)
    heap: list[tuple[float, float, int, int]] = []
    heapq.heappush(
        heap,
        (
            _heuristic_km(sr, sc, gr, gc, km_per_row=km_per_row, km_per_col_min=km_per_col_min),
            0.0,
            sr,
            sc,
        ),
    )

    while heap:
        _, cur_g, r, c = heapq.heappop(heap)
        if closed[r, c]:
            continue
        closed[r, c] = True
        if (r, c) == (gr, gc):
            break

        cur = (r, c)
        par = parent.get(cur, cur)
        for rr, cc in _neighbors(r, c, h, w):
            if closed[rr, cc] or blocked[rr, cc]:
                continue

            nxt = (rr, cc)
            best_parent = cur
            cand = cur_g + _transition_cost(
                from_rc=cur,
                to_rc=nxt,
                geo=geo,
                caution=caution,
                ais_norm=ais_norm,
                caution_penalty=caution_penalty,
                corridor_reward=corridor_reward,
                near_blocked=near_blocked,
                near_blocked_penalty=near_blocked_penalty,
                uncertainty_penalty=uncertainty_penalty,
                risk_penalty=risk_penalty,
            )
            if par != cur:
                turn_deg = _turn_angle_deg(par, cur, nxt)
                cand += float(cand * turn_weight * (turn_deg / 180.0))

            if par != cur and _line_of_sight(par, nxt, blocked):
                par_cost = gscore[par[0], par[1]] + _transition_cost_segment(
                    from_rc=par,
                    to_rc=nxt,
                    geo=geo,
                    caution=caution,
                    ais_norm=ais_norm,
                    caution_penalty=caution_penalty,
                    corridor_reward=corridor_reward,
                    near_blocked=near_blocked,
                    near_blocked_penalty=near_blocked_penalty,
                    uncertainty_penalty=uncertainty_penalty,
                    risk_penalty=risk_penalty,
                )
                grand = parent.get(par, par)
                if grand != par:
                    turn_deg = _turn_angle_deg(grand, par, nxt)
                    par_cost += float(par_cost * turn_weight * (turn_deg / 180.0))
                if par_cost < cand:
                    cand = par_cost
                    best_parent = par

            if cand < gscore[rr, cc]:
                gscore[rr, cc] = cand
                parent[nxt] = best_parent
                came_from[nxt] = best_parent
                hval = _heuristic_km(rr, cc, gr, gc, km_per_row=km_per_row, km_per_col_min=km_per_col_min)
                heapq.heappush(heap, (cand + hval, cand, rr, cc))

    if not np.isfinite(gscore[gr, gc]):
        raise PlanningError("No feasible route found under current blocked constraints.")
    return _reconstruct_path(came_from, (gr, gc))


def _heading_bin_from_delta(dr: int, dc: int, heading_bins: int) -> int:
    angle = math.atan2(float(dr), float(dc))
    step = (2.0 * math.pi) / float(heading_bins)
    return int(round((angle % (2.0 * math.pi)) / step)) % heading_bins


def _run_hybrid_astar(
    *,
    geo,
    blocked: np.ndarray,
    caution: np.ndarray,
    ais_norm: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    km_per_row: float,
    km_per_col_min: float,
    caution_penalty: float,
    corridor_reward: float,
    near_blocked: np.ndarray | None = None,
    near_blocked_penalty: float = 0.0,
    uncertainty_penalty: np.ndarray | None = None,
    risk_penalty: np.ndarray | None = None,
    heading_bins: int = 8,
) -> list[tuple[int, int]]:
    """Lightweight hybrid A*: heading-aware A* over (r,c,heading) with 8-neighbor expansions."""
    h, w = blocked.shape
    sr, sc = start
    gr, gc = goal

    init_heading = _heading_bin_from_delta(gr - sr, gc - sc, heading_bins)
    gscore = np.full((h, w, heading_bins), np.inf, dtype=np.float64)
    closed = np.zeros((h, w, heading_bins), dtype=bool)
    came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    goal_state: tuple[int, int, int] | None = None

    gscore[sr, sc, init_heading] = 0.0
    heap: list[tuple[float, float, int, int, int]] = []
    heapq.heappush(
        heap,
        (
            _heuristic_km(sr, sc, gr, gc, km_per_row=km_per_row, km_per_col_min=km_per_col_min),
            0.0,
            sr,
            sc,
            init_heading,
        ),
    )
    turn_weight = 0.04
    max_iter = h * w * heading_bins * 2
    it = 0

    while heap:
        if it > max_iter:
            raise PlanningError("Hybrid A* exceeded iteration budget.")
        it += 1
        _, cur_g, r, c, heading = heapq.heappop(heap)
        if closed[r, c, heading]:
            continue
        closed[r, c, heading] = True
        if (r, c) == (gr, gc):
            goal_state = (r, c, heading)
            break

        # Local 8-neighbor expansions keep runtime bounded and guarantee feasibility close to grid A*.
        for rr, cc in _neighbors(r, c, h, w):
            if blocked[rr, cc]:
                continue
            if not _line_of_sight((r, c), (rr, cc), blocked):
                continue
            next_heading = _heading_bin_from_delta(rr - r, cc - c, heading_bins)
            diff = abs(next_heading - heading)
            diff = min(diff, heading_bins - diff)
            seg = _transition_cost(
                from_rc=(r, c),
                to_rc=(rr, cc),
                geo=geo,
                caution=caution,
                ais_norm=ais_norm,
                caution_penalty=caution_penalty,
                corridor_reward=corridor_reward,
                near_blocked=near_blocked,
                near_blocked_penalty=near_blocked_penalty,
                uncertainty_penalty=uncertainty_penalty,
                risk_penalty=risk_penalty,
            )
            turn_penalty = seg * turn_weight * float(diff)
            cand = cur_g + seg + turn_penalty
            if cand < gscore[rr, cc, next_heading]:
                gscore[rr, cc, next_heading] = cand
                state = (rr, cc, next_heading)
                came_from[state] = (r, c, heading)
                hval = _heuristic_km(rr, cc, gr, gc, km_per_row=km_per_row, km_per_col_min=km_per_col_min)
                heapq.heappush(heap, (cand + hval, cand, rr, cc, next_heading))

    if goal_state is None:
        raise PlanningError("No feasible route found under current blocked constraints.")

    # Reconstruct (r,c,heading) and project to grid cells.
    path_states = [goal_state]
    cur = goal_state
    while cur in came_from:
        cur = came_from[cur]
        path_states.append(cur)
    path_states.reverse()

    cells: list[tuple[int, int]] = []
    for rr, cc, _ in path_states:
        if not cells or cells[-1] != (rr, cc):
            cells.append((rr, cc))
    if not cells or cells[0] != start:
        cells.insert(0, start)
    return cells


def _run_dstar_lite_static(
    *,
    geo,
    blocked: np.ndarray,
    caution: np.ndarray,
    ais_norm: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    caution_penalty: float,
    corridor_reward: float,
    near_blocked: np.ndarray | None = None,
    near_blocked_penalty: float = 0.0,
    uncertainty_penalty: np.ndarray | None = None,
    risk_penalty: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Static-grid backward potential field + greedy extraction."""
    g_to_goal = _compute_cost_to_goal(
        geo=geo,
        blocked=blocked,
        caution=caution,
        ais_norm=ais_norm,
        goal=goal,
        caution_penalty=caution_penalty,
        corridor_reward=corridor_reward,
        near_blocked=near_blocked,
        near_blocked_penalty=near_blocked_penalty,
        uncertainty_penalty=uncertainty_penalty,
        risk_penalty=risk_penalty,
    )
    h, w = blocked.shape
    gr, gc = goal
    sr, sc = start

    if not np.isfinite(g_to_goal[sr, sc]):
        raise PlanningError("No feasible route found under current blocked constraints.")

    cells = [(sr, sc)]
    visited = {cells[0]}
    cur = (sr, sc)
    max_steps = h * w
    for _ in range(max_steps):
        if cur == (gr, gc):
            return cells
        r, c = cur
        best_next = None
        best_cost = np.inf
        for rr, cc in _neighbors(r, c, h, w):
            if blocked[rr, cc]:
                continue
            tail = g_to_goal[rr, cc]
            if not np.isfinite(tail):
                continue
            cand = _transition_cost(
                from_rc=cur,
                to_rc=(rr, cc),
                geo=geo,
                caution=caution,
                ais_norm=ais_norm,
                caution_penalty=caution_penalty,
                corridor_reward=corridor_reward,
                near_blocked=near_blocked,
                near_blocked_penalty=near_blocked_penalty,
                uncertainty_penalty=uncertainty_penalty,
                risk_penalty=risk_penalty,
            ) + tail
            if cand < best_cost:
                best_cost = cand
                best_next = (rr, cc)
        if best_next is None:
            break
        if best_next in visited:
            # Guard against local loops from numeric ties.
            raise PlanningError("Planner entered a loop while extracting path.")
        cells.append(best_next)
        visited.add(best_next)
        cur = best_next

    raise PlanningError("No feasible route found under current blocked constraints.")


def _compute_cost_to_goal(
    *,
    geo,
    blocked: np.ndarray,
    caution: np.ndarray,
    ais_norm: np.ndarray,
    goal: tuple[int, int],
    caution_penalty: float,
    corridor_reward: float,
    near_blocked: np.ndarray | None = None,
    near_blocked_penalty: float = 0.0,
    uncertainty_penalty: np.ndarray | None = None,
    risk_penalty: np.ndarray | None = None,
) -> np.ndarray:
    h, w = blocked.shape
    gr, gc = goal

    g_to_goal = np.full((h, w), np.inf, dtype=np.float64)
    g_to_goal[gr, gc] = 0.0
    heap: list[tuple[float, int, int]] = [(0.0, gr, gc)]

    while heap:
        dist_u, r, c = heapq.heappop(heap)
        if dist_u > g_to_goal[r, c]:
            continue
        for pr, pc in _neighbors(r, c, h, w):
            if blocked[pr, pc]:
                continue
            # Backward relaxation uses forward transition cost predecessor->current.
            cand = dist_u + _transition_cost(
                from_rc=(pr, pc),
                to_rc=(r, c),
                geo=geo,
                caution=caution,
                ais_norm=ais_norm,
                caution_penalty=caution_penalty,
                corridor_reward=corridor_reward,
                near_blocked=near_blocked,
                near_blocked_penalty=near_blocked_penalty,
                uncertainty_penalty=uncertainty_penalty,
                risk_penalty=risk_penalty,
            )
            if cand < g_to_goal[pr, pc]:
                g_to_goal[pr, pc] = cand
                heapq.heappush(heap, (cand, pr, pc))
    return g_to_goal


class _DStarLiteIncremental:
    """D* Lite on 8-neighbor grid with incremental blocked-cell updates."""

    def __init__(
        self,
        *,
        geo,
        blocked: np.ndarray,
        caution: np.ndarray,
        ais_norm: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        km_per_row: float,
        km_per_col_min: float,
        caution_penalty: float,
        corridor_reward: float,
        near_blocked: np.ndarray | None = None,
        near_blocked_penalty: float = 0.0,
        uncertainty_penalty: np.ndarray | None = None,
        risk_penalty: np.ndarray | None = None,
    ) -> None:
        self.geo = geo
        self.blocked = blocked.copy()
        self.caution = caution.copy()
        self.ais_norm = ais_norm.copy()
        self.start = start
        self.goal = goal
        self.last = start
        self.km = 0.0
        self.km_per_row = km_per_row
        self.km_per_col_min = km_per_col_min
        self.caution_penalty = caution_penalty
        self.corridor_reward = corridor_reward
        self.near_blocked = near_blocked.copy() if near_blocked is not None else _build_near_blocked_mask(self.blocked)
        self.near_blocked_penalty = near_blocked_penalty
        self.uncertainty_penalty = uncertainty_penalty.copy() if uncertainty_penalty is not None else None
        self.risk_penalty = risk_penalty.copy() if risk_penalty is not None else None

        h, w = blocked.shape
        self.h = h
        self.w = w
        g_to_goal = _compute_cost_to_goal(
            geo=self.geo,
            blocked=self.blocked,
            caution=self.caution,
            ais_norm=self.ais_norm,
            goal=self.goal,
            caution_penalty=self.caution_penalty,
            corridor_reward=self.corridor_reward,
            near_blocked=self.near_blocked,
            near_blocked_penalty=self.near_blocked_penalty,
            uncertainty_penalty=self.uncertainty_penalty,
            risk_penalty=self.risk_penalty,
        )
        self.g = g_to_goal.copy()
        self.rhs = g_to_goal.copy()

        self.open_heap: list[tuple[float, float, int, int]] = []
        self.open_best: dict[tuple[int, int], tuple[float, float]] = {}
        self._needs_replan = False

    @staticmethod
    def _lex_lt(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return a[0] < b[0] - 1e-12 or (abs(a[0] - b[0]) <= 1e-12 and a[1] < b[1] - 1e-12)

    def _heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        ar, ac = a
        br, bc = b
        return _heuristic_km(
            ar,
            ac,
            br,
            bc,
            km_per_row=self.km_per_row,
            km_per_col_min=self.km_per_col_min,
        )

    def _calculate_key(self, u: tuple[int, int]) -> tuple[float, float]:
        r, c = u
        g_rhs = min(self.g[r, c], self.rhs[r, c])
        return (g_rhs + self._heuristic(self.start, u) + self.km, g_rhs)

    def _push(self, u: tuple[int, int], key: tuple[float, float]) -> None:
        self.open_best[u] = key
        heapq.heappush(self.open_heap, (key[0], key[1], u[0], u[1]))

    def _peek(self) -> tuple[tuple[float, float], tuple[int, int]] | None:
        while self.open_heap:
            k0, k1, r, c = self.open_heap[0]
            u = (r, c)
            best = self.open_best.get(u)
            if best is None or abs(best[0] - k0) > 1e-12 or abs(best[1] - k1) > 1e-12:
                heapq.heappop(self.open_heap)
                continue
            return (k0, k1), u
        return None

    def _pop(self) -> tuple[tuple[float, float], tuple[int, int]] | None:
        while self.open_heap:
            k0, k1, r, c = heapq.heappop(self.open_heap)
            u = (r, c)
            best = self.open_best.get(u)
            if best is None or abs(best[0] - k0) > 1e-12 or abs(best[1] - k1) > 1e-12:
                continue
            del self.open_best[u]
            return (k0, k1), u
        return None

    def _cost(self, u: tuple[int, int], v: tuple[int, int]) -> float:
        ur, uc = u
        vr, vc = v
        if self.blocked[ur, uc] or self.blocked[vr, vc]:
            return math.inf
        return _transition_cost(
            from_rc=u,
            to_rc=v,
            geo=self.geo,
            caution=self.caution,
            ais_norm=self.ais_norm,
            caution_penalty=self.caution_penalty,
            corridor_reward=self.corridor_reward,
            near_blocked=self.near_blocked,
            near_blocked_penalty=self.near_blocked_penalty,
            uncertainty_penalty=self.uncertainty_penalty,
            risk_penalty=self.risk_penalty,
        )

    def _update_vertex(self, u: tuple[int, int]) -> None:
        ur, uc = u
        if u != self.goal:
            best = math.inf
            if not self.blocked[ur, uc]:
                for vr, vc in _neighbors(ur, uc, self.h, self.w):
                    cand = self._cost(u, (vr, vc)) + self.g[vr, vc]
                    if cand < best:
                        best = cand
            self.rhs[ur, uc] = best

        if u in self.open_best:
            del self.open_best[u]
        g_val = self.g[ur, uc]
        rhs_val = self.rhs[ur, uc]
        inconsistent = (np.isfinite(g_val) != np.isfinite(rhs_val)) or (
            np.isfinite(g_val) and np.isfinite(rhs_val) and abs(g_val - rhs_val) > 1e-12
        )
        if inconsistent:
            self._push(u, self._calculate_key(u))

    def compute_shortest_path(self) -> None:
        max_iter = self.h * self.w * 64
        it = 0
        sr, sc = self.start
        while True:
            top = self._peek()
            top_key = top[0] if top is not None else (math.inf, math.inf)
            start_key = self._calculate_key(self.start)
            if not (self._lex_lt(top_key, start_key) or abs(self.rhs[sr, sc] - self.g[sr, sc]) > 1e-12):
                return
            if it > max_iter:
                raise PlanningError("D* Lite exceeded iteration budget while replanning.")
            it += 1
            popped = self._pop()
            if popped is None:
                return
            old_key, u = popped
            new_key = self._calculate_key(u)
            if self._lex_lt(old_key, new_key):
                self._push(u, new_key)
                continue

            ur, uc = u
            if self.g[ur, uc] > self.rhs[ur, uc]:
                self.g[ur, uc] = self.rhs[ur, uc]
                for pr, pc in _neighbors(ur, uc, self.h, self.w):
                    self._update_vertex((pr, pc))
            else:
                self.g[ur, uc] = math.inf
                self._update_vertex(u)
                for pr, pc in _neighbors(ur, uc, self.h, self.w):
                    self._update_vertex((pr, pc))

    def update_blocked(
        self,
        new_blocked: np.ndarray,
        changed_cells: np.ndarray,
        *,
        auto_compute: bool = True,
    ) -> dict[str, int]:
        if new_blocked.shape != (self.h, self.w):
            raise PlanningError("Dynamic update shape mismatch in D* Lite planner.")
        if changed_cells.size == 0:
            return {"changed_cells": 0, "changed_edge_count": 0, "touched_vertices": 0}
        np.copyto(self.blocked, new_blocked)
        self.near_blocked = _build_near_blocked_mask(self.blocked)
        touched: set[tuple[int, int]] = set()
        for rc in changed_cells:
            r = int(rc[0])
            c = int(rc[1])
            touched.add((r, c))
            for pr, pc in _neighbors(r, c, self.h, self.w):
                touched.add((pr, pc))
        for u in touched:
            self._update_vertex(u)
        self._needs_replan = True
        if auto_compute:
            self.compute_shortest_path()
            self._needs_replan = False
        return {
            "changed_cells": int(changed_cells.shape[0]),
            "changed_edge_count": int(_count_changed_directed_edges(changed_cells, h=self.h, w=self.w)),
            "touched_vertices": int(len(touched)),
        }

    def update_cost_maps(
        self,
        *,
        caution: np.ndarray,
        ais_norm: np.ndarray,
        uncertainty_penalty: np.ndarray | None,
        risk_penalty: np.ndarray | None,
        changed_mask: np.ndarray | None = None,
        auto_compute: bool = True,
    ) -> dict[str, int]:
        if caution.shape != (self.h, self.w) or ais_norm.shape != (self.h, self.w):
            raise PlanningError("Dynamic cost-map update shape mismatch in D* Lite planner.")
        if changed_mask is None:
            changed_mask = (
                (self.caution != caution)
                | (np.abs(self.ais_norm - ais_norm) > 1e-6)
                | _optional_array_diff_mask(self.uncertainty_penalty, uncertainty_penalty, (self.h, self.w), atol=1e-6)
                | _optional_array_diff_mask(self.risk_penalty, risk_penalty, (self.h, self.w), atol=1e-6)
            )
        changed_cells = np.argwhere(changed_mask)
        if changed_cells.size == 0:
            return {"changed_cells": 0, "changed_edge_count": 0, "touched_vertices": 0}

        self.caution = caution.copy()
        self.ais_norm = ais_norm.copy()
        self.uncertainty_penalty = uncertainty_penalty.copy() if uncertainty_penalty is not None else None
        self.risk_penalty = risk_penalty.copy() if risk_penalty is not None else None

        touched: set[tuple[int, int]] = set()
        for rc in changed_cells:
            r = int(rc[0])
            c = int(rc[1])
            touched.add((r, c))
            for pr, pc in _neighbors(r, c, self.h, self.w):
                touched.add((pr, pc))
        for u in touched:
            self._update_vertex(u)
        self._needs_replan = True
        if auto_compute:
            self.compute_shortest_path()
            self._needs_replan = False
        return {
            "changed_cells": int(changed_cells.shape[0]),
            "changed_edge_count": int(_count_changed_directed_edges(changed_cells, h=self.h, w=self.w)),
            "touched_vertices": int(len(touched)),
        }

    def move_start(self, new_start: tuple[int, int], *, auto_compute: bool = True) -> None:
        if new_start == self.start:
            return
        self.km += self._heuristic(self.last, new_start)
        self.last = new_start
        self.start = new_start
        self._needs_replan = True
        if auto_compute:
            self.compute_shortest_path()
            self._needs_replan = False

    def sync_if_needed(self) -> None:
        if self._needs_replan:
            self.compute_shortest_path()
            self._needs_replan = False

    def extract_path(self, max_steps: int | None = None) -> list[tuple[int, int]]:
        self.sync_if_needed()
        max_steps = max_steps or self.h * self.w
        if self.blocked[self.start[0], self.start[1]]:
            raise PlanningError("Current start is blocked in D* Lite extraction.")
        if not np.isfinite(self.rhs[self.start[0], self.start[1]]):
            raise PlanningError("No feasible route found under current blocked constraints.")

        path = [self.start]
        cur = self.start
        visited = {cur}
        for _ in range(max_steps):
            if cur == self.goal:
                return path
            cr, cc = cur
            best_next = None
            best_cost = math.inf
            for nr, nc in _neighbors(cr, cc, self.h, self.w):
                cand = self._cost(cur, (nr, nc)) + self.g[nr, nc]
                if cand < best_cost:
                    best_cost = cand
                    best_next = (nr, nc)
            if best_next is None or not np.isfinite(best_cost):
                raise PlanningError("No feasible route found under current blocked constraints.")
            if best_next in visited:
                raise PlanningError("D* Lite path extraction loop detected.")
            path.append(best_next)
            visited.add(best_next)
            cur = best_next
        raise PlanningError("D* Lite failed to reach goal within step budget.")


def plan_grid_route_dynamic(
    *,
    settings: Settings,
    timestamps: list[str],
    start: tuple[float, float],
    goal: tuple[float, float],
    model_version: str,
    corridor_bias: float,
    caution_mode: str,
    smoothing: bool,
    blocked_sources: list[str],
    planner: str = "dstar_lite",
    advance_steps: int = 12,
    uncertainty_uplift: bool = True,
    uncertainty_uplift_scale: float = 1.0,
    risk_mode: str = "balanced",
    risk_weight_scale: float = 1.0,
    risk_constraint_mode: str = "none",
    risk_budget: float = 1.0,
    confidence_level: float = 0.90,
    dynamic_replan_mode: str = "on_event",
    replan_blocked_ratio: float = 0.002,
    replan_risk_spike: float = 0.05,
    replan_corridor_min: float = 0.05,
    replan_max_skip_steps: int = 2,
    dynamic_risk_switch_enabled: bool = False,
    dynamic_risk_budget_km: float = 1.0,
    dynamic_risk_warn_ratio: float = 0.7,
    dynamic_risk_hard_ratio: float = 1.0,
    dynamic_risk_warn_mode: str = "conservative",
    dynamic_risk_hard_mode: str = "conservative",
    dynamic_risk_switch_min_interval: int = 1,
) -> PlanResult:
    if len(timestamps) < 2:
        raise PlanningError("Dynamic replanning requires at least 2 timestamps.")
    if advance_steps < 1:
        raise PlanningError("advance_steps must be >= 1")
    valid_caution_modes = {"tie_breaker", "budget", "minimize", "strict"}
    if caution_mode not in valid_caution_modes:
        raise PlanningError(f"Unsupported caution_mode={caution_mode}, expected one of {sorted(valid_caution_modes)}")
    valid_risk_modes = {"conservative", "balanced", "aggressive"}
    if risk_mode not in valid_risk_modes:
        raise PlanningError(f"Unsupported risk_mode={risk_mode}, expected one of {sorted(valid_risk_modes)}")
    if dynamic_risk_warn_mode not in valid_risk_modes:
        raise PlanningError(
            f"Unsupported dynamic_risk_warn_mode={dynamic_risk_warn_mode}, expected one of {sorted(valid_risk_modes)}"
        )
    if dynamic_risk_hard_mode not in valid_risk_modes:
        raise PlanningError(
            f"Unsupported dynamic_risk_hard_mode={dynamic_risk_hard_mode}, expected one of {sorted(valid_risk_modes)}"
        )
    if dynamic_risk_warn_ratio < 0.0:
        raise PlanningError("dynamic_risk_warn_ratio must be >= 0")
    if dynamic_risk_hard_ratio < 0.0:
        raise PlanningError("dynamic_risk_hard_ratio must be >= 0")
    if dynamic_risk_hard_ratio < dynamic_risk_warn_ratio:
        raise PlanningError("dynamic_risk_hard_ratio must be >= dynamic_risk_warn_ratio")
    if dynamic_risk_budget_km < 0.0:
        raise PlanningError("dynamic_risk_budget_km must be >= 0")
    if dynamic_risk_switch_min_interval < 1:
        raise PlanningError("dynamic_risk_switch_min_interval must be >= 1")
    valid_constraint_modes = {"none", "chance", "cvar"}
    if risk_constraint_mode not in valid_constraint_modes:
        raise PlanningError(
            f"Unsupported risk_constraint_mode={risk_constraint_mode}, expected one of {sorted(valid_constraint_modes)}"
        )
    valid_replan_modes = {"always", "on_event"}
    if dynamic_replan_mode not in valid_replan_modes:
        raise PlanningError(
            f"Unsupported dynamic_replan_mode={dynamic_replan_mode}, expected one of {sorted(valid_replan_modes)}"
        )

    risk_modes_for_state = [risk_mode]
    if dynamic_risk_switch_enabled:
        risk_modes_for_state.extend([dynamic_risk_warn_mode, dynamic_risk_hard_mode])
    risk_modes_for_state = sorted(set(str(m).strip().lower() for m in risk_modes_for_state if str(m).strip()))
    runtime_notes: list[str] = []

    def _load_state_timed(ts: str) -> tuple[GridState, dict[str, float | str]]:
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        state_loaded = _load_grid_state(
            settings=settings,
            timestamp=ts,
            model_version=model_version,
            blocked_sources=blocked_sources,
            caution_mode=caution_mode,
            uncertainty_uplift=uncertainty_uplift,
            uncertainty_uplift_scale=uncertainty_uplift_scale,
            risk_mode=risk_mode,
            risk_weight_scale=risk_weight_scale,
            risk_modes_to_load=risk_modes_for_state,
        )
        return state_loaded, {
            "timestamp": str(ts),
            "wall_ms": (time.perf_counter() - t0) * 1000.0,
            "cpu_ms": (time.process_time() - cpu0) * 1000.0,
        }

    state_load_workers = 1
    state_load_mode = "sequential"
    load_pairs: list[tuple[GridState, dict[str, float | str]]] = []
    state_load_t0 = time.perf_counter()
    if len(timestamps) >= 3:
        state_load_workers = min(4, len(timestamps))
        state_load_mode = "parallel"
        try:
            with ThreadPoolExecutor(max_workers=state_load_workers) as executor:
                futures = [executor.submit(_load_state_timed, ts) for ts in timestamps]
                load_pairs = [f.result() for f in futures]
        except Exception as exc:
            state_load_mode = "parallel_fallback"
            state_load_workers = 1
            runtime_notes.append(f"parallel_state_load_fallback:{exc}")
            load_pairs = [_load_state_timed(ts) for ts in timestamps]
    else:
        load_pairs = [_load_state_timed(ts) for ts in timestamps]

    states = [pair[0] for pair in load_pairs]
    state_load_rows = [pair[1] for pair in load_pairs]
    state_load_wall_ms = (time.perf_counter() - state_load_t0) * 1000.0
    h, w = states[0].blocked.shape
    for state in states[1:]:
        if state.blocked.shape != (h, w):
            raise PlanningError("Dynamic replanning requires aligned grid shape across timestamps.")

    planner_key = planner.strip().lower()
    is_astar = planner_key in {"astar", "a_star"}
    is_dstar_incremental = planner_key in {"dstar_lite", "dstar-lite", "dstar"}
    is_dstar_recompute = planner_key in {"dstar_lite_recompute", "dstar_recompute", "dstar-recompute", "dstar_full"}
    is_dstar = is_dstar_incremental or is_dstar_recompute
    is_any_angle = planner_key in {"any_angle", "any-angle", "theta", "theta_star", "thetastar"}
    is_hybrid = planner_key in {"hybrid_astar", "hybrid-a*", "hybrid", "hybrid_a_star"}
    if is_astar:
        planner_label = "astar"
    elif is_dstar_incremental:
        planner_label = "dstar_lite"
    elif is_dstar_recompute:
        planner_label = "dstar_lite_recompute"
    elif is_any_angle:
        planner_label = "any_angle"
    elif is_hybrid:
        planner_label = "hybrid_astar"
    else:
        planner_label = planner_key
    if not (is_astar or is_dstar or is_any_angle or is_hybrid):
        raise PlanningError(
            "Unsupported planner={planner}, expected one of astar, dstar_lite, "
            "dstar_lite_recompute, any_angle, hybrid_astar".format(planner=planner)
        )

    caution_penalty, corridor_reward, near_blocked_penalty = _policy_scalars(caution_mode, corridor_bias)
    km_per_row, km_per_col_min = _grid_resolution_km(states[0].bounds, h, w)

    sr0, sc0, s_inside = states[0].geo.latlon_to_rc(start[0], start[1])
    gr0, gc0, g_inside = states[0].geo.latlon_to_rc(goal[0], goal[1])
    if not s_inside:
        b = states[0].bounds
        raise PlanningError(
            f"Point out of grid bounds: lat={start[0]}, lon={start[1]}, "
            f"bounds=({b.lat_min},{b.lat_max},{b.lon_min},{b.lon_max})"
        )
    if not g_inside:
        b = states[0].bounds
        raise PlanningError(
            f"Point out of grid bounds: lat={goal[0]}, lon={goal[1]}, "
            f"bounds=({b.lat_min},{b.lat_max},{b.lon_min},{b.lon_max})"
        )

    start_rc = _nearest_unblocked((sr0, sc0), states[0].blocked, states[0].free_cells)
    goal_rc = _nearest_unblocked((gr0, gc0), states[0].blocked, states[0].free_cells)
    start_adjusted = start_rc != (sr0, sc0)
    goal_adjusted = goal_rc != (gr0, gc0)
    s_lat0, s_lon0 = states[0].geo.rc_to_latlon(sr0, sc0)
    s_lat1, s_lon1 = states[0].geo.rc_to_latlon(*start_rc)
    g_lat0, g_lon0 = states[0].geo.rc_to_latlon(gr0, gc0)
    g_lat1, g_lon1 = states[0].geo.rc_to_latlon(*goal_rc)
    start_adjust_km = haversine_km(s_lat0, s_lon0, s_lat1, s_lon1)
    goal_adjust_km = haversine_km(g_lat0, g_lon0, g_lat1, g_lon1)

    executed_coords: list[list[float]] = []
    executed_coords_raw: list[list[float]] = []
    executed_points = 0
    executed_caution_points = 0
    executed_adjacent_blocked_points = 0
    corridor_vals: list[float] = []
    total_distance_km = 0.0
    base_distance_km = 0.0
    caution_len_km = 0.0
    cost_caution_extra_km = 0.0
    cost_corridor_discount_km = 0.0
    cost_uncertainty_extra_km = 0.0
    cost_risk_extra_km = 0.0
    replans: list[dict] = []
    route_cells_executed = 0
    dynamic_notes: list[str] = list(runtime_notes)
    uncertainty_vals: list[float] = []
    risk_vals: list[float] = []
    step_runtime_samples_ms: list[float] = []
    step_update_samples_ms: list[float] = []
    step_planner_samples_ms: list[float] = []
    step_metrics_samples_ms: list[float] = []
    step_cpu_samples_ms: list[float] = []
    step_memory_samples_mb: list[float] = []
    runtime_mem_peak_mb = _memory_usage_mb()
    runtime_cache_hits = 0
    runtime_cache_misses = 0
    metrics_cache: dict[tuple, dict] = {}
    metrics_cache_order: list[tuple] = []
    full_graph_edges = _grid_directed_edge_count(h, w)
    cached_cells: list[tuple[int, int]] = []
    trigger_events: list[dict] = []
    last_replan_step = -1
    execution_log: list[dict] = []
    cumulative_replan_runtime_ms = 0.0
    active_risk_mode = str(risk_mode).strip().lower()
    prev_step_risk_penalty: np.ndarray | None = None
    last_risk_switch_step = -10_000
    risk_switch_events: list[dict] = []
    risk_mode_timeline: list[dict] = []
    risk_budget_protection_steps = 0

    current_start = start_rc
    current_goal = goal_rc

    dstar: _DStarLiteIncremental | None = None
    current_cells: list[tuple[int, int]] = []
    last_state_used = states[0]

    def _collect_path_metrics_cached(
        *,
        cache_key: tuple,
        cells: list[tuple[int, int]],
        state_now: GridState,
        risk_penalty_now: np.ndarray | None,
    ) -> dict:
        nonlocal runtime_cache_hits, runtime_cache_misses
        if cache_key in metrics_cache:
            runtime_cache_hits += 1
            return metrics_cache[cache_key]
        runtime_cache_misses += 1
        near_blocked = state_now.near_blocked if state_now.near_blocked is not None else _build_near_blocked_mask(state_now.blocked)
        metrics = _collect_path_metrics(
            cells=cells,
            geo=state_now.geo,
            caution=state_now.caution,
            ais_norm=state_now.ais_norm,
            near_blocked=near_blocked,
            caution_penalty=caution_penalty,
            corridor_reward=corridor_reward,
            uncertainty_penalty=state_now.uncertainty_penalty,
            risk_penalty=risk_penalty_now,
        )
        metrics_cache[cache_key] = metrics
        metrics_cache_order.append(cache_key)
        if len(metrics_cache_order) > PATH_METRICS_CACHE_MAX:
            drop_key = metrics_cache_order.pop(0)
            metrics_cache.pop(drop_key, None)
        return metrics

    for step_idx, state in enumerate(states):
        last_state_used = state
        step_t0 = time.perf_counter()
        step_cpu_t0 = time.process_time()
        update_t0 = time.perf_counter()
        step_update_mode = "init" if step_idx == 0 else "none"
        step_planner_runtime_ms = 0.0
        step_metrics_runtime_ms = 0.0
        step_changed = 0
        step_changed_blocked = 0
        step_changed_caution = 0
        step_changed_ais = 0
        step_changed_uncertainty = 0
        step_changed_risk = 0
        step_changed_edge_count = 0
        step_touched_vertices = 0
        goal_now = _nearest_unblocked(current_goal, state.blocked, state.free_cells)
        if goal_now != current_goal:
            current_goal = goal_now
            dstar = None
            dynamic_notes.append(f"goal_adjusted_at_step_{step_idx}")

        prev_state = states[step_idx - 1] if step_idx > 0 else None
        risk_stage, target_risk_mode, risk_budget_usage = _dynamic_risk_stage_and_target_mode(
            enabled=bool(dynamic_risk_switch_enabled),
            cumulative_risk_extra_km=float(cost_risk_extra_km),
            budget_km=float(dynamic_risk_budget_km),
            base_mode=str(risk_mode),
            warn_mode=str(dynamic_risk_warn_mode),
            hard_mode=str(dynamic_risk_hard_mode),
            warn_ratio=float(dynamic_risk_warn_ratio),
            hard_ratio=float(dynamic_risk_hard_ratio),
        )
        switched_this_step = False
        previous_risk_mode = active_risk_mode
        if (
            target_risk_mode != active_risk_mode
            and (step_idx - last_risk_switch_step) >= int(dynamic_risk_switch_min_interval)
        ):
            before_penalty = _state_risk_penalty_for_mode(state, active_risk_mode)
            after_penalty = _state_risk_penalty_for_mode(state, target_risk_mode)
            eval_cells = cached_cells if len(cached_cells) >= 2 else []
            risk_before = _mean_penalty_on_cells(before_penalty, eval_cells)
            risk_after = _mean_penalty_on_cells(after_penalty, eval_cells)
            switch_delta = float(risk_before - risk_after)
            active_risk_mode = target_risk_mode
            last_risk_switch_step = step_idx
            switched_this_step = True
            risk_switch_events.append(
                {
                    "step": int(step_idx),
                    "timestamp": str(state.timestamp),
                    "from_mode": str(previous_risk_mode),
                    "to_mode": str(target_risk_mode),
                    "budget_stage": str(risk_stage),
                    "budget_usage": round(float(risk_budget_usage), 6),
                    "estimated_risk_before": round(float(risk_before), 6),
                    "estimated_risk_after": round(float(risk_after), 6),
                    "estimated_risk_gain": round(float(switch_delta), 6),
                    "cooldown_steps": int(dynamic_risk_switch_min_interval),
                }
            )
            dynamic_notes.append(
                "risk_mode_switch_step_{step}: {before}->{after} stage={stage} usage={usage:.4f}".format(
                    step=step_idx,
                    before=previous_risk_mode,
                    after=target_risk_mode,
                    stage=risk_stage,
                    usage=float(risk_budget_usage),
                )
            )

        step_risk_penalty = _state_risk_penalty_for_mode(state, active_risk_mode)
        step_risk_meta = _state_risk_meta_for_mode(state, active_risk_mode)
        risk_mode_timeline.append(
            {
                "step": int(step_idx),
                "timestamp": str(state.timestamp),
                "risk_mode": str(active_risk_mode),
                "risk_stage": str(risk_stage),
                "risk_budget_usage": round(float(risk_budget_usage), 6),
                "switched": bool(switched_this_step),
            }
        )
        if prev_state is not None:
            diff_prev_blocked = prev_state.blocked != state.blocked
            diff_prev_caution = prev_state.caution != state.caution
            diff_prev_ais = np.abs(prev_state.ais_norm - state.ais_norm) > 1e-6
            diff_prev_unc = _optional_array_diff_mask(prev_state.uncertainty_penalty, state.uncertainty_penalty, (h, w))
            diff_prev_risk = _optional_array_diff_mask(prev_step_risk_penalty, step_risk_penalty, (h, w))
            diff_prev_union = diff_prev_blocked | diff_prev_caution | diff_prev_ais | diff_prev_unc | diff_prev_risk
            step_changed_blocked = int(np.count_nonzero(diff_prev_blocked))
            step_changed_caution = int(np.count_nonzero(diff_prev_caution))
            step_changed_ais = int(np.count_nonzero(diff_prev_ais))
            step_changed_uncertainty = int(np.count_nonzero(diff_prev_unc))
            step_changed_risk = int(np.count_nonzero(diff_prev_risk))
            step_changed = int(np.count_nonzero(diff_prev_union))
            step_changed_edge_count = _count_changed_directed_edges(np.argwhere(diff_prev_union), h=h, w=w)

        cached_cells = _align_cached_path(cached_cells, current_start)
        trigger_reasons: list[str] = []
        trigger_metrics: dict[str, float | int] = {
            "blocked_change_ratio": round(float(step_changed_blocked / max(1, (h * w))), 6),
            "cached_path_blocked_ratio": 0.0,
            "cached_corridor_alignment": 0.0,
            "cached_risk_delta": 0.0,
            "risk_budget_usage": round(float(risk_budget_usage), 6),
        }
        if step_idx == 0:
            trigger_reasons = ["initial_step"]
        elif dynamic_replan_mode == "always":
            trigger_reasons = ["always_mode"]
        else:
            if not cached_cells or len(cached_cells) < 2:
                trigger_reasons.append("no_cached_path")
            if step_changed_blocked > 0:
                blocked_ratio = float(step_changed_blocked / max(1, (h * w)))
                if blocked_ratio >= float(replan_blocked_ratio):
                    trigger_reasons.append("blocked_change_ratio")
            if cached_cells:
                blocked_ratio_on_path = _cells_blocked_ratio(cached_cells, state.blocked)
                trigger_metrics["cached_path_blocked_ratio"] = round(float(blocked_ratio_on_path), 6)
                if blocked_ratio_on_path > 0.0:
                    trigger_reasons.append("path_blocked")
                cached_metrics_t0 = time.perf_counter()
                metrics_cached = _collect_path_metrics_cached(
                    cache_key=(str(state.timestamp), str(active_risk_mode), tuple(cached_cells)),
                    cells=cached_cells,
                    state_now=state,
                    risk_penalty_now=step_risk_penalty,
                )
                step_metrics_runtime_ms += (time.perf_counter() - cached_metrics_t0) * 1000.0
                corridor_now = float(metrics_cached.get("corridor_alignment", 0.0))
                trigger_metrics["cached_corridor_alignment"] = round(corridor_now, 6)
                if corridor_now < float(replan_corridor_min):
                    trigger_reasons.append("corridor_mismatch")
                risk_now = _mean_penalty_on_cells(step_risk_penalty, cached_cells)
                risk_prev = _mean_penalty_on_cells(prev_step_risk_penalty, cached_cells)
                risk_delta = float(risk_now - risk_prev)
                trigger_metrics["cached_risk_delta"] = round(risk_delta, 6)
                if risk_delta >= float(replan_risk_spike):
                    trigger_reasons.append("risk_spike")
            if last_replan_step >= 0 and (step_idx - last_replan_step) >= int(replan_max_skip_steps):
                trigger_reasons.append("max_skip_steps")
        if switched_this_step:
            trigger_reasons.append("risk_mode_switch")
        if risk_stage == "hard":
            trigger_reasons.append("risk_budget_pressure")

        should_replan = bool(trigger_reasons)
        trigger_events.append(
            {
                "step": int(step_idx),
                "timestamp": str(state.timestamp),
                "mode": str(dynamic_replan_mode),
                "should_replan": bool(should_replan),
                "reasons": list(trigger_reasons),
                "metrics": trigger_metrics,
            }
        )

        planner_t0 = time.perf_counter()
        if not should_replan and len(cached_cells) >= 2:
            current_cells = cached_cells
            step_update_mode = "reuse"
        elif is_astar:
            current_cells = _run_astar(
                geo=state.geo,
                blocked=state.blocked,
                caution=state.caution,
                ais_norm=state.ais_norm,
                start=current_start,
                goal=current_goal,
                km_per_row=km_per_row,
                km_per_col_min=km_per_col_min,
                caution_penalty=caution_penalty,
                corridor_reward=corridor_reward,
                near_blocked=state.near_blocked,
                near_blocked_penalty=near_blocked_penalty,
                uncertainty_penalty=state.uncertainty_penalty,
                risk_penalty=step_risk_penalty,
            )
            step_update_mode = "rebuild"
            step_changed_edge_count = full_graph_edges
            step_touched_vertices = h * w
        elif is_any_angle:
            current_cells = _run_theta_star(
                geo=state.geo,
                blocked=state.blocked,
                caution=state.caution,
                ais_norm=state.ais_norm,
                start=current_start,
                goal=current_goal,
                km_per_row=km_per_row,
                km_per_col_min=km_per_col_min,
                caution_penalty=caution_penalty,
                corridor_reward=corridor_reward,
                near_blocked=state.near_blocked,
                near_blocked_penalty=near_blocked_penalty,
                uncertainty_penalty=state.uncertainty_penalty,
                risk_penalty=step_risk_penalty,
            )
            step_update_mode = "rebuild"
            step_changed_edge_count = full_graph_edges
            step_touched_vertices = h * w
        elif is_hybrid:
            current_cells = _run_hybrid_astar(
                geo=state.geo,
                blocked=state.blocked,
                caution=state.caution,
                ais_norm=state.ais_norm,
                start=current_start,
                goal=current_goal,
                km_per_row=km_per_row,
                km_per_col_min=km_per_col_min,
                caution_penalty=caution_penalty,
                corridor_reward=corridor_reward,
                near_blocked=state.near_blocked,
                near_blocked_penalty=near_blocked_penalty,
                uncertainty_penalty=state.uncertainty_penalty,
                risk_penalty=step_risk_penalty,
            )
            step_update_mode = "rebuild"
            step_changed_edge_count = full_graph_edges
            step_touched_vertices = h * w
        else:
            if dstar is None:
                dstar = _DStarLiteIncremental(
                    geo=state.geo,
                    blocked=state.blocked,
                    caution=state.caution,
                    ais_norm=state.ais_norm,
                    start=current_start,
                    goal=current_goal,
                    km_per_row=km_per_row,
                    km_per_col_min=km_per_col_min,
                    caution_penalty=caution_penalty,
                    corridor_reward=corridor_reward,
                    near_blocked=state.near_blocked,
                    near_blocked_penalty=near_blocked_penalty,
                    uncertainty_penalty=state.uncertainty_penalty,
                    risk_penalty=step_risk_penalty,
                )
                step_update_mode = "rebuild"
                step_changed_edge_count = full_graph_edges
                step_touched_vertices = h * w
            else:
                diff_blocked = dstar.blocked != state.blocked
                diff_caution = dstar.caution != state.caution
                diff_ais = np.abs(dstar.ais_norm - state.ais_norm) > 1e-6
                diff_unc = _optional_array_diff_mask(dstar.uncertainty_penalty, state.uncertainty_penalty, (h, w))
                diff_risk = _optional_array_diff_mask(dstar.risk_penalty, step_risk_penalty, (h, w))
                diff_cost = diff_caution | diff_ais | diff_unc | diff_risk
                diff_union = diff_blocked | diff_cost

                blocked_changed = int(np.count_nonzero(diff_blocked))
                caution_changed = int(np.count_nonzero(diff_caution))
                ais_changed = int(np.count_nonzero(diff_ais))
                uncertainty_changed = int(np.count_nonzero(diff_unc))
                risk_changed = int(np.count_nonzero(diff_risk))
                step_changed = int(np.count_nonzero(diff_union))
                step_changed_blocked = blocked_changed
                step_changed_caution = caution_changed
                step_changed_ais = ais_changed
                step_changed_uncertainty = uncertainty_changed
                step_changed_risk = risk_changed
                step_changed_edge_count = _count_changed_directed_edges(np.argwhere(diff_union), h=h, w=w)
                change_ratio = float(step_changed) / float(h * w) if (h * w) > 0 else 0.0
                incremental_limit = max(DSTAR_INCREMENTAL_CHANGED_MIN_CELLS, int(h * w * DSTAR_INCREMENTAL_CHANGED_RATIO))
                can_incremental = step_changed > 0 and step_changed <= incremental_limit and not is_dstar_recompute
                if can_incremental:
                    touched_vertices = 0
                    if blocked_changed:
                        blocked_stats = dstar.update_blocked(state.blocked, np.argwhere(diff_blocked), auto_compute=False)
                        touched_vertices += int(blocked_stats.get("touched_vertices", 0))
                    if caution_changed or ais_changed or uncertainty_changed or risk_changed:
                        cost_stats = dstar.update_cost_maps(
                            caution=state.caution,
                            ais_norm=state.ais_norm,
                            uncertainty_penalty=state.uncertainty_penalty,
                            risk_penalty=step_risk_penalty,
                            changed_mask=diff_cost,
                            auto_compute=False,
                        )
                        touched_vertices += int(cost_stats.get("touched_vertices", 0))
                    step_touched_vertices = touched_vertices
                    step_update_mode = "incremental"
                elif step_changed:
                    dstar = _DStarLiteIncremental(
                        geo=state.geo,
                        blocked=state.blocked,
                        caution=state.caution,
                        ais_norm=state.ais_norm,
                        start=current_start,
                        goal=current_goal,
                        km_per_row=km_per_row,
                        km_per_col_min=km_per_col_min,
                        caution_penalty=caution_penalty,
                        corridor_reward=corridor_reward,
                        near_blocked=state.near_blocked,
                        near_blocked_penalty=near_blocked_penalty,
                        uncertainty_penalty=state.uncertainty_penalty,
                        risk_penalty=step_risk_penalty,
                    )
                    step_update_mode = "rebuild"
                    step_changed_edge_count = full_graph_edges
                    step_touched_vertices = h * w
                    dynamic_notes.append(
                        "rebuild_step_{step}: blocked_changed={blocked_changed} "
                        "caution_changed={caution_changed} ais_changed={ais_changed} "
                        "uncertainty_changed={unc_changed} risk_changed={risk_changed} ratio={ratio:.4f}".format(
                            step=step_idx,
                            blocked_changed=blocked_changed,
                            caution_changed=caution_changed,
                            ais_changed=ais_changed,
                            unc_changed=uncertainty_changed,
                            risk_changed=risk_changed,
                            ratio=change_ratio,
                        )
                    )
                if current_goal != dstar.goal:
                    dstar = _DStarLiteIncremental(
                        geo=state.geo,
                        blocked=state.blocked,
                        caution=state.caution,
                        ais_norm=state.ais_norm,
                        start=current_start,
                        goal=current_goal,
                        km_per_row=km_per_row,
                        km_per_col_min=km_per_col_min,
                        caution_penalty=caution_penalty,
                        corridor_reward=corridor_reward,
                        near_blocked=state.near_blocked,
                        near_blocked_penalty=near_blocked_penalty,
                        uncertainty_penalty=state.uncertainty_penalty,
                        risk_penalty=step_risk_penalty,
                    )
                    step_update_mode = "rebuild"
                    step_changed_edge_count = full_graph_edges
                    step_touched_vertices = h * w
                if dstar.start != current_start:
                    dstar.move_start(current_start, auto_compute=False)
                dstar.sync_if_needed()
            current_cells = dstar.extract_path()
        step_planner_runtime_ms = (time.perf_counter() - planner_t0) * 1000.0

        if should_replan:
            last_replan_step = step_idx
        step_update_runtime_ms = (time.perf_counter() - update_t0) * 1000.0

        raw_points = len(current_cells)
        move_cells = current_cells
        smoothed_fallback = False
        smoothed_fallback_reason = ""
        max_turn_limit_deg = 110.0
        if smoothing:
            if is_any_angle or is_hybrid:
                move_cells = _smooth_cells_los_constrained(
                    current_cells,
                    state.blocked,
                    max_turn_deg=max_turn_limit_deg,
                )
            else:
                move_cells = _smooth_cells_los(current_cells, state.blocked)
        if not _is_path_feasible(move_cells, state.blocked):
            move_cells = current_cells
            smoothed_fallback = True
            smoothed_fallback_reason = "smoothed_path_crosses_blocked"
        smoothed_points = len(move_cells)
        if len(move_cells) < 2:
            break

        move_edges = min(advance_steps, len(move_cells) - 1) if step_idx < len(states) - 1 else len(move_cells) - 1
        if dynamic_risk_switch_enabled and risk_stage == "hard" and step_idx < len(states) - 1:
            guarded_edges = max(1, min(move_edges, max(1, int(math.ceil(float(advance_steps) * 0.5)))))
            if guarded_edges < move_edges:
                move_edges = guarded_edges
                risk_budget_protection_steps += 1
        step_distance_km = 0.0
        step_caution_km = 0.0
        step_caution_extra_km = 0.0
        step_corridor_discount_km = 0.0
        step_uncertainty_extra_km = 0.0
        step_risk_extra_km = 0.0
        step_samples = 0
        step_caution_hits = 0
        step_near_blocked_hits = 0
        metric_cells = move_cells[: move_edges + 1]
        step_metrics_t0 = time.perf_counter()
        step_metrics = _collect_path_metrics_cached(
            cache_key=(str(state.timestamp), str(active_risk_mode), tuple(metric_cells)),
            cells=metric_cells,
            state_now=state,
            risk_penalty_now=step_risk_penalty,
        )
        step_metrics_runtime_ms += (time.perf_counter() - step_metrics_t0) * 1000.0
        step_distance_km = float(step_metrics["distance_km"])
        step_caution_km = float(step_metrics["caution_len_km"])
        step_caution_extra_km = float(step_metrics["cost_caution_extra_km"])
        step_corridor_discount_km = float(step_metrics["cost_corridor_discount_km"])
        step_uncertainty_extra_km = float(step_metrics["cost_uncertainty_extra_km"])
        step_risk_extra_km = float(step_metrics["cost_risk_extra_km"])
        step_samples = int(step_metrics["sample_count"])
        step_caution_hits = int(step_metrics["caution_hits"])
        step_near_blocked_hits = int(step_metrics["near_blocked_hits"])
        corridor_vals.extend(step_metrics["corridor_vals"])
        uncertainty_vals.extend(step_metrics["uncertainty_penalty_vals"])
        risk_vals.extend(step_metrics["risk_penalty_vals"])
        candidate_coords: list[list[float]] = []
        for rr, cc in move_cells:
            cand_lat, cand_lon = state.geo.rc_to_latlon(rr, cc)
            candidate_coords.append([cand_lon, cand_lat])

        segment_coords: list[list[float]] = []
        for idx in range(1, move_edges + 1):
            pr, pc = move_cells[idx - 1]
            cr, cc = move_cells[idx]
            plat, plon = state.geo.rc_to_latlon(pr, pc)
            clat, clon = state.geo.rc_to_latlon(cr, cc)
            if not segment_coords:
                segment_coords.append([plon, plat])
            segment_coords.append([clon, clat])
            if not executed_coords:
                executed_coords.append([plon, plat])
            executed_coords.append([clon, clat])
        move_edges_raw = min(move_edges, max(0, len(current_cells) - 1))
        for idx in range(1, move_edges_raw + 1):
            pr, pc = current_cells[idx - 1]
            cr, cc = current_cells[idx]
            plat, plon = state.geo.rc_to_latlon(pr, pc)
            clat, clon = state.geo.rc_to_latlon(cr, cc)
            if not executed_coords_raw:
                executed_coords_raw.append([plon, plat])
            executed_coords_raw.append([clon, clat])

        executed_points += step_samples
        executed_caution_points += step_caution_hits
        executed_adjacent_blocked_points += step_near_blocked_hits

        total_distance_km += step_distance_km
        base_distance_km += step_distance_km
        caution_len_km += step_caution_km
        cost_caution_extra_km += step_caution_extra_km
        cost_corridor_discount_km += step_corridor_discount_km
        cost_uncertainty_extra_km += step_uncertainty_extra_km
        cost_risk_extra_km += step_risk_extra_km
        route_cells_executed += move_edges

        moved_to = move_cells[move_edges]
        current_start = _nearest_unblocked(moved_to, state.blocked, state.free_cells)
        cached_cells = _align_cached_path(move_cells[move_edges:], current_start)
        step_runtime_ms = (time.perf_counter() - step_t0) * 1000.0
        step_cpu_ms = (time.process_time() - step_cpu_t0) * 1000.0
        step_memory_mb = _memory_usage_mb()
        runtime_mem_peak_mb = max(float(runtime_mem_peak_mb), float(step_memory_mb))
        cumulative_replan_runtime_ms += float(step_runtime_ms)
        step_runtime_samples_ms.append(float(step_runtime_ms))
        step_update_samples_ms.append(float(step_update_runtime_ms))
        step_planner_samples_ms.append(float(step_planner_runtime_ms))
        step_metrics_samples_ms.append(float(step_metrics_runtime_ms))
        step_cpu_samples_ms.append(float(step_cpu_ms))
        step_memory_samples_mb.append(float(step_memory_mb))
        step_effective_cost_km = float(
            step_distance_km
            + step_caution_extra_km
            + step_uncertainty_extra_km
            + step_risk_extra_km
            - step_corridor_discount_km
        )
        replans.append(
            {
                "step": step_idx,
                "timestamp": state.timestamp,
                "runtime_ms": round(float(step_runtime_ms), 3),
                "cpu_runtime_ms": round(float(step_cpu_ms), 3),
                "planner_runtime_ms": round(float(step_planner_runtime_ms), 3),
                "metrics_runtime_ms": round(float(step_metrics_runtime_ms), 3),
                "update_runtime_ms": round(float(step_update_runtime_ms), 3),
                "memory_mb": round(float(step_memory_mb), 3),
                "raw_points": raw_points,
                "smoothed_points": smoothed_points,
                "moved_edges": move_edges,
                "moved_distance_km": round(float(step_distance_km), 3),
                "step_caution_len_km": round(float(step_caution_km), 3),
                "step_risk_extra_km": round(float(step_risk_extra_km), 3),
                "step_effective_cost_km": round(float(step_effective_cost_km), 3),
                "cumulative_distance_km": round(float(total_distance_km), 3),
                "cumulative_risk_extra_km": round(float(cost_risk_extra_km), 3),
                "cumulative_replan_runtime_ms": round(float(cumulative_replan_runtime_ms), 3),
                "changed_blocked_cells": int(step_changed_blocked if step_idx > 0 else 0),
                "changed_caution_cells": int(step_changed_caution if step_idx > 0 else 0),
                "changed_ais_cells": int(step_changed_ais if step_idx > 0 else 0),
                "changed_uncertainty_cells": int(step_changed_uncertainty if step_idx > 0 else 0),
                "changed_risk_cells": int(step_changed_risk if step_idx > 0 else 0),
                "changed_cells_total": int(step_changed if step_idx > 0 else 0),
                "changed_edge_count": int(step_changed_edge_count if step_idx > 0 else full_graph_edges),
                "touched_vertices": int(step_touched_vertices),
                "update_mode": step_update_mode,
                "triggered_replan": bool(should_replan),
                "trigger_reasons": list(trigger_reasons),
                "risk_mode": str(active_risk_mode),
                "risk_budget_stage": str(risk_stage),
                "risk_budget_usage": round(float(risk_budget_usage), 6),
                "risk_mode_switched": bool(switched_this_step),
                "risk_lambda": round(float(step_risk_meta.get("risk_lambda", 0.0)), 4),
                "smoothing_feasible": not smoothed_fallback,
                "smoothing_fallback_reason": smoothed_fallback_reason,
            }
        )
        execution_log.append(
            {
                "step": int(step_idx),
                "timestamp": str(state.timestamp),
                "update_mode": str(step_update_mode),
                "triggered_replan": bool(should_replan),
                "trigger_reasons": list(trigger_reasons),
                "risk_mode": str(active_risk_mode),
                "risk_budget_stage": str(risk_stage),
                "risk_budget_usage": round(float(risk_budget_usage), 6),
                "risk_mode_switched": bool(switched_this_step),
                "risk_lambda": round(float(step_risk_meta.get("risk_lambda", 0.0)), 4),
                "moved_edges": int(move_edges),
                "moved_distance_km": round(float(step_distance_km), 3),
                "step_effective_cost_km": round(float(step_effective_cost_km), 3),
                "step_risk_extra_km": round(float(step_risk_extra_km), 3),
                "cumulative_distance_km": round(float(total_distance_km), 3),
                "cumulative_risk_extra_km": round(float(cost_risk_extra_km), 3),
                "replan_runtime_ms": round(float(step_runtime_ms), 3),
                "replan_cpu_runtime_ms": round(float(step_cpu_ms), 3),
                "planner_runtime_ms": round(float(step_planner_runtime_ms), 3),
                "metrics_runtime_ms": round(float(step_metrics_runtime_ms), 3),
                "memory_mb": round(float(step_memory_mb), 3),
                "cumulative_replan_runtime_ms": round(float(cumulative_replan_runtime_ms), 3),
                "segment_coordinates": segment_coords,
                "segment_start": segment_coords[0] if segment_coords else [],
                "segment_end": segment_coords[-1] if segment_coords else [],
                "candidate_coordinates": candidate_coords,
            }
        )

        if current_start == current_goal:
            prev_step_risk_penalty = step_risk_penalty
            break
        prev_step_risk_penalty = step_risk_penalty

    incremental_steps = int(sum(1 for r in replans if r.get("update_mode") == "incremental"))
    rebuild_steps = int(sum(1 for r in replans if r.get("update_mode") == "rebuild"))
    if is_dstar_incremental:
        planner_label = "dstar_lite_incremental"
    elif is_dstar_recompute:
        planner_label = "dstar_lite_recompute"
    elif is_any_angle:
        planner_label = "any_angle_recompute"
    elif is_hybrid:
        planner_label = "hybrid_astar_recompute"
    else:
        planner_label = "astar_recompute"
    uncertainty_meta = next(
        (s.uncertainty_meta for s in reversed(states) if isinstance(s.uncertainty_meta, dict) and s.uncertainty_meta),
        {"enabled": bool(uncertainty_uplift), "applied": False},
    )
    risk_meta = _state_risk_meta_for_mode(last_state_used, active_risk_mode)
    state_load_wall_samples = [float(row.get("wall_ms", 0.0)) for row in state_load_rows]
    state_load_cpu_samples = [float(row.get("cpu_ms", 0.0)) for row in state_load_rows]
    step_wall_stats = _metric_summary(step_runtime_samples_ms)
    step_cpu_stats = _metric_summary(step_cpu_samples_ms)
    step_update_stats = _metric_summary(step_update_samples_ms)
    step_planner_stats = _metric_summary(step_planner_samples_ms)
    step_metrics_stats = _metric_summary(step_metrics_samples_ms)
    state_load_wall_stats = _metric_summary(state_load_wall_samples)
    state_load_cpu_stats = _metric_summary(state_load_cpu_samples)
    cache_total = int(runtime_cache_hits + runtime_cache_misses)
    cache_hit_ratio = float(runtime_cache_hits / max(1, cache_total))
    runtime_cpu_wall_ratio = float(step_cpu_stats["total"] / max(1e-6, step_wall_stats["total"]))
    runtime_monitor = {
        "state_load_mode": str(state_load_mode),
        "state_load_workers": int(state_load_workers),
        "state_load_wall_ms_total": round(float(state_load_wall_ms), 3),
        "state_load_wall_ms_sum": round(float(state_load_wall_stats["total"]), 3),
        "state_load_wall_ms_mean": round(float(state_load_wall_stats["mean"]), 3),
        "state_load_wall_ms_p90": round(float(state_load_wall_stats["p90"]), 3),
        "state_load_cpu_ms_total": round(float(state_load_cpu_stats["total"]), 3),
        "state_load_cpu_ms_mean": round(float(state_load_cpu_stats["mean"]), 3),
        "step_wall_ms_total": round(float(step_wall_stats["total"]), 3),
        "step_wall_ms_mean": round(float(step_wall_stats["mean"]), 3),
        "step_wall_ms_p90": round(float(step_wall_stats["p90"]), 3),
        "step_wall_ms_max": round(float(step_wall_stats["max"]), 3),
        "step_cpu_ms_total": round(float(step_cpu_stats["total"]), 3),
        "step_cpu_ms_mean": round(float(step_cpu_stats["mean"]), 3),
        "step_cpu_ms_p90": round(float(step_cpu_stats["p90"]), 3),
        "step_update_ms_total": round(float(step_update_stats["total"]), 3),
        "step_update_ms_mean": round(float(step_update_stats["mean"]), 3),
        "step_planner_ms_total": round(float(step_planner_stats["total"]), 3),
        "step_planner_ms_mean": round(float(step_planner_stats["mean"]), 3),
        "step_metrics_ms_total": round(float(step_metrics_stats["total"]), 3),
        "step_metrics_ms_mean": round(float(step_metrics_stats["mean"]), 3),
        "cpu_wall_ratio": round(float(runtime_cpu_wall_ratio), 4),
        "path_metrics_cache_hits": int(runtime_cache_hits),
        "path_metrics_cache_misses": int(runtime_cache_misses),
        "path_metrics_cache_hit_ratio": round(float(cache_hit_ratio), 6),
        "memory_peak_mb": round(float(runtime_mem_peak_mb), 3),
        "memory_last_mb": round(float(step_memory_samples_mb[-1] if step_memory_samples_mb else runtime_mem_peak_mb), 3),
    }
    explain = {
        "distance_km": round(float(total_distance_km), 3),
        "distance_nm": round(float(total_distance_km) * 0.539957, 3),
        "caution_len_km": round(float(caution_len_km), 3),
        "corridor_alignment": round(float(np.mean(corridor_vals) if corridor_vals else 0.0), 3),
        "corridor_alignment_p50": round(float(np.percentile(corridor_vals, 50)) if corridor_vals else 0.0, 3),
        "corridor_alignment_p90": round(float(np.percentile(corridor_vals, 90)) if corridor_vals else 0.0, 3),
        "caution_mode": caution_mode,
        "effective_caution_penalty": round(float(caution_penalty), 4),
        "effective_corridor_reward": round(float(corridor_reward), 4),
        "effective_near_blocked_penalty": round(float(near_blocked_penalty), 4),
        "route_cost_base_km": round(float(base_distance_km), 3),
        "route_cost_caution_extra_km": round(float(cost_caution_extra_km), 3),
        "route_cost_corridor_discount_km": round(float(cost_corridor_discount_km), 3),
        "route_cost_uncertainty_extra_km": round(float(cost_uncertainty_extra_km), 3),
        "route_cost_risk_extra_km": round(float(cost_risk_extra_km), 3),
        "route_cost_effective_km": round(
            float(
                base_distance_km
                + cost_caution_extra_km
                + cost_uncertainty_extra_km
                + cost_risk_extra_km
                - cost_corridor_discount_km
            ),
            3,
        ),
        "risk_mode": str(risk_meta.get("risk_mode", risk_mode)),
        "risk_layer": str(risk_meta.get("risk_layer", "risk_mean")),
        "risk_lambda": round(float(risk_meta.get("risk_lambda", 0.0)), 4),
        "risk_penalty_mean": round(float(np.mean(risk_vals) if risk_vals else 0.0), 4),
        "risk_penalty_p90": round(float(np.percentile(risk_vals, 90)) if risk_vals else 0.0, 4),
        "risk_meta": risk_meta,
        "uncertainty_uplift_enabled": bool(uncertainty_uplift),
        "uncertainty_uplift_scale": round(float(uncertainty_meta.get("uplift_scale", 0.0)), 4),
        "uncertainty_threshold": round(float(uncertainty_meta.get("threshold", 0.0)), 4),
        "uncertainty_temperature": round(float(uncertainty_meta.get("temperature", 1.0)), 4),
        "uncertainty_profile_available": bool(uncertainty_meta.get("profile_available", False)),
        "uncertainty_penalty_mean": round(float(np.mean(uncertainty_vals) if uncertainty_vals else 0.0), 4),
        "uncertainty_penalty_p90": round(float(np.percentile(uncertainty_vals, 90)) if uncertainty_vals else 0.0, 4),
        "uncertainty_calibration": uncertainty_meta,
        "smoothing": bool(smoothing),
        "start_adjusted": start_adjusted,
        "goal_adjusted": goal_adjusted,
        "start_adjust_km": round(float(start_adjust_km), 3),
        "goal_adjust_km": round(float(goal_adjust_km), 3),
        "planner": planner_label,
        "dynamic_replans": replans,
        "dynamic_replan_mode": str(dynamic_replan_mode),
        "dynamic_trigger_events": trigger_events,
        "dynamic_trigger_count": int(sum(1 for e in trigger_events if bool(e.get("should_replan")))),
        "dynamic_trigger_thresholds": {
            "replan_blocked_ratio": round(float(replan_blocked_ratio), 6),
            "replan_risk_spike": round(float(replan_risk_spike), 6),
            "replan_corridor_min": round(float(replan_corridor_min), 6),
            "replan_max_skip_steps": int(replan_max_skip_steps),
        },
        "dynamic_advance_steps": int(advance_steps),
        "dynamic_timestamps": timestamps,
        "dynamic_notes": dynamic_notes,
        "dynamic_risk_switch_enabled": bool(dynamic_risk_switch_enabled),
        "dynamic_risk_budget_km": round(float(dynamic_risk_budget_km), 4),
        "dynamic_risk_budget_warn_ratio": round(float(dynamic_risk_warn_ratio), 4),
        "dynamic_risk_budget_hard_ratio": round(float(dynamic_risk_hard_ratio), 4),
        "dynamic_risk_mode_base": str(risk_mode),
        "dynamic_risk_mode_warn": str(dynamic_risk_warn_mode),
        "dynamic_risk_mode_hard": str(dynamic_risk_hard_mode),
        "dynamic_risk_mode_active_last": str(active_risk_mode),
        "dynamic_risk_switch_min_interval": int(dynamic_risk_switch_min_interval),
        "dynamic_risk_switch_events": risk_switch_events,
        "dynamic_risk_mode_timeline": risk_mode_timeline,
        "dynamic_risk_switch_count": int(len(risk_switch_events)),
        "dynamic_risk_budget_usage_final": round(
            float(cost_risk_extra_km / max(1e-8, float(dynamic_risk_budget_km))) if dynamic_risk_switch_enabled else 0.0,
            6,
        ),
        "dynamic_risk_budget_protection_steps": int(risk_budget_protection_steps),
        "dynamic_risk_switch_gain_total": round(
            float(sum(float(e.get("estimated_risk_gain", 0.0)) for e in risk_switch_events)),
            6,
        ),
        "dynamic_risk_switch_gain_mean": round(
            float(
                sum(float(e.get("estimated_risk_gain", 0.0)) for e in risk_switch_events)
                / max(1, len(risk_switch_events))
            ),
            6,
        ),
        "dynamic_incremental_steps": incremental_steps,
        "dynamic_rebuild_steps": rebuild_steps,
        "dynamic_reuse_steps": int(sum(1 for r in replans if r.get("update_mode") == "reuse")),
        "dynamic_simulation_mode": "advance_replan_loop",
        "dynamic_execution_log": execution_log,
        "dynamic_execution_steps": int(len(execution_log)),
        "dynamic_cumulative_distance_km": round(float(total_distance_km), 3),
        "dynamic_cumulative_risk_extra_km": round(float(cost_risk_extra_km), 3),
        "dynamic_cumulative_replan_runtime_ms": round(float(cumulative_replan_runtime_ms), 3),
        "dynamic_cumulative_effective_cost_km": round(
            float(
                base_distance_km
                + cost_caution_extra_km
                + cost_uncertainty_extra_km
                + cost_risk_extra_km
                - cost_corridor_discount_km
            ),
            3,
        ),
        "dynamic_replay_ready": True,
        "dynamic_incremental_threshold_ratio": round(float(DSTAR_INCREMENTAL_CHANGED_RATIO), 4),
        "executed_edges": int(route_cells_executed),
        "blocked_ratio_last": round(float(states[-1].blocked.mean()), 4),
        "adjacent_blocked_ratio": round(float(executed_adjacent_blocked_points / max(1, executed_points)), 4),
        "caution_cell_ratio": round(float(executed_caution_points / max(1, executed_points)), 4),
        "dynamic_runtime_monitor": runtime_monitor,
        "dynamic_runtime_state_load_rows": state_load_rows,
        "dynamic_path_metrics_cache_hits": int(runtime_cache_hits),
        "dynamic_path_metrics_cache_misses": int(runtime_cache_misses),
        "dynamic_path_metrics_cache_hit_ratio": round(float(cache_hit_ratio), 6),
        "dynamic_runtime_cpu_wall_ratio": round(float(runtime_cpu_wall_ratio), 4),
        "dynamic_runtime_memory_peak_mb": round(float(runtime_mem_peak_mb), 3),
        "dynamic_runtime_step_wall_ms_p90": round(float(step_wall_stats["p90"]), 3),
        "dynamic_runtime_step_wall_ms_mean": round(float(step_wall_stats["mean"]), 3),
        "dynamic_runtime_step_update_ms_mean": round(float(step_update_stats["mean"]), 3),
        "replan_runtime_ms_total": round(float(sum(float(r.get("runtime_ms", 0.0)) for r in replans)), 3),
        "replan_runtime_ms_mean": round(
            float(sum(float(r.get("runtime_ms", 0.0)) for r in replans) / max(1, len(replans))),
            3,
        ),
        "replan_update_runtime_ms_total": round(float(sum(float(r.get("update_runtime_ms", 0.0)) for r in replans)), 3),
        "replan_update_runtime_ms_mean": round(
            float(sum(float(r.get("update_runtime_ms", 0.0)) for r in replans) / max(1, len(replans))),
            3,
        ),
        "replan_changed_cells_total": int(sum(int(r.get("changed_cells_total", 0)) for r in replans)),
        "replan_changed_edges_total": int(sum(int(r.get("changed_edge_count", 0)) for r in replans)),
        "replan_full_graph_edges": int(full_graph_edges),
    }

    risk_lambda = float(risk_meta.get("risk_lambda", 0.0))
    raw_risk_values = _risk_raw_values_from_penalty_values(risk_vals, risk_lambda=risk_lambda)
    constraint = _evaluate_risk_constraint(
        mode=risk_constraint_mode,
        budget=float(risk_budget),
        confidence_level=float(confidence_level),
        raw_risk_values=raw_risk_values,
    )
    explain.update(
        {
            "risk_constraint_mode": constraint["mode"],
            "risk_budget": round(float(constraint["budget"]), 4),
            "risk_confidence_level": round(float(constraint["confidence_level"]), 4),
            "risk_constraint_metric_name": str(constraint["metric_name"]),
            "risk_constraint_metric": round(float(constraint["metric"]), 6),
            "risk_budget_usage": round(float(constraint["usage"]), 6),
            "risk_constraint_satisfied": bool(constraint["satisfied"]),
            "risk_constraint_sample_count": int(constraint["sample_count"]),
            "risk_constraint_tail_count": int(constraint["tail_count"]),
        }
    )
    if constraint["mode"] != "none" and not bool(constraint["satisfied"]):
        raise PlanningError(
            "Risk constraint violated in dynamic planning: "
            f"{constraint['metric_name']}={constraint['metric']:.4f} > budget={constraint['budget']:.4f}"
        )

    if is_any_angle or is_hybrid:
        display_coords = executed_coords
    else:
        display_coords = _build_display_coordinates(executed_coords)
    route_geojson = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": executed_coords},
        "properties": {
            **explain,
            "display_coordinates": display_coords,
            "raw_coordinates": executed_coords_raw,
            "feasible_smoothed_coordinates": executed_coords,
        },
    }
    return PlanResult(route_geojson=route_geojson, explain=explain)


def plan_grid_route(
    *,
    settings: Settings,
    timestamp: str,
    start: tuple[float, float],
    goal: tuple[float, float],
    model_version: str,
    corridor_bias: float,
    caution_mode: str,
    smoothing: bool,
    blocked_sources: list[str],
    planner: str = "astar",
    uncertainty_uplift: bool = True,
    uncertainty_uplift_scale: float = 1.0,
    risk_mode: str = "balanced",
    risk_weight_scale: float = 1.0,
    risk_constraint_mode: str = "none",
    risk_budget: float = 1.0,
    confidence_level: float = 0.90,
) -> PlanResult:
    valid_caution_modes = {"tie_breaker", "budget", "minimize", "strict"}
    if caution_mode not in valid_caution_modes:
        raise PlanningError(f"Unsupported caution_mode={caution_mode}, expected one of {sorted(valid_caution_modes)}")
    valid_risk_modes = {"conservative", "balanced", "aggressive"}
    if risk_mode not in valid_risk_modes:
        raise PlanningError(f"Unsupported risk_mode={risk_mode}, expected one of {sorted(valid_risk_modes)}")
    valid_constraint_modes = {"none", "chance", "cvar"}
    if risk_constraint_mode not in valid_constraint_modes:
        raise PlanningError(
            f"Unsupported risk_constraint_mode={risk_constraint_mode}, expected one of {sorted(valid_constraint_modes)}"
        )

    ann_dir = settings.annotation_pack_root / timestamp
    blocked_path = ann_dir / "blocked_mask.npy"
    if not blocked_path.exists():
        raise PlanningError(f"blocked_mask missing for timestamp={timestamp}: {blocked_path}")
    blocked_bathy = np.load(blocked_path).astype(np.uint8) > 0
    h, w = blocked_bathy.shape
    geo = load_grid_geo(settings, timestamp=timestamp, shape=(h, w))

    pred_path = settings.pred_root / model_version / f"{timestamp}.npy"
    if ("unet_blocked" in blocked_sources or caution_mode == "tie_breaker") and not pred_path.exists():
        run_unet_inference(
            settings=settings,
            timestamp=timestamp,
            model_version=model_version,
            output_path=pred_path,
        )
    unet_pred = np.load(pred_path).astype(np.uint8) if pred_path.exists() else np.zeros((h, w), dtype=np.uint8)
    if unet_pred.shape != (h, w):
        raise PlanningError(f"Pred shape mismatch for timestamp={timestamp}: {unet_pred.shape} vs {(h, w)}")

    blocked = blocked_bathy.copy()
    if "unet_blocked" in blocked_sources:
        blocked |= unet_pred == 2
    if "unet_caution" in blocked_sources or caution_mode == "strict":
        blocked |= unet_pred == 1
    free_cells = np.argwhere(~blocked)
    if free_cells.size == 0:
        raise PlanningError("No navigable cells available after blocked-mask fusion.")
    caution = unet_pred == 1
    ais = _load_ais_heatmap(settings, timestamp=timestamp, shape=(h, w))
    ais = np.clip(ais, 0.0, None)
    ais_max = float(ais.max())
    ais_norm = ais / ais_max if ais_max > 1e-8 else np.zeros_like(ais)
    unc_cal, unc_penalty, unc_meta = _load_uncertainty_penalty(
        settings=settings,
        timestamp=timestamp,
        model_version=model_version,
        shape=(h, w),
        enabled=uncertainty_uplift,
        uplift_scale_factor=uncertainty_uplift_scale,
    )
    risk_penalty, risk_meta = _load_risk_penalty(
        settings=settings,
        timestamp=timestamp,
        model_version=model_version,
        shape=(h, w),
        risk_mode=risk_mode,
        risk_weight_scale=risk_weight_scale,
    )

    bounds = GridBounds(
        lat_min=float(geo.bounds.lat_min),
        lat_max=float(geo.bounds.lat_max),
        lon_min=float(geo.bounds.lon_min),
        lon_max=float(geo.bounds.lon_max),
    )
    sr0, sc0, s_inside = geo.latlon_to_rc(start[0], start[1])
    gr0, gc0, g_inside = geo.latlon_to_rc(goal[0], goal[1])
    if not s_inside:
        raise PlanningError(
            f"Point out of grid bounds: lat={start[0]}, lon={start[1]}, "
            f"bounds=({bounds.lat_min},{bounds.lat_max},{bounds.lon_min},{bounds.lon_max})"
        )
    if not g_inside:
        raise PlanningError(
            f"Point out of grid bounds: lat={goal[0]}, lon={goal[1]}, "
            f"bounds=({bounds.lat_min},{bounds.lat_max},{bounds.lon_min},{bounds.lon_max})"
        )
    s_rc = (sr0, sc0)
    g_rc = (gr0, gc0)
    s_rc_adj = _nearest_unblocked(s_rc, blocked, free_cells)
    g_rc_adj = _nearest_unblocked(g_rc, blocked, free_cells)

    km_per_row, km_per_col_min = _grid_resolution_km(bounds, h, w)
    caution_penalty, corridor_reward, near_blocked_penalty = _policy_scalars(caution_mode, corridor_bias)
    near_blocked = _build_near_blocked_mask(blocked)

    sr, sc = s_rc_adj
    gr, gc = g_rc_adj
    planner_key = planner.strip().lower()
    is_astar = planner_key in {"astar", "a_star"}
    is_dstar = planner_key in {
        "dstar_lite",
        "dstar-lite",
        "dstar",
        "dstar_lite_recompute",
        "dstar_recompute",
        "dstar-recompute",
        "dstar_full",
    }
    is_any_angle = planner_key in {"any_angle", "any-angle", "theta", "theta_star", "thetastar"}
    is_hybrid = planner_key in {"hybrid_astar", "hybrid-a*", "hybrid", "hybrid_a_star"}
    if is_astar:
        planner_label = "astar"
    elif is_dstar:
        planner_label = "dstar_lite" if planner_key in {"dstar_lite", "dstar-lite", "dstar"} else "dstar_lite_recompute"
    elif is_any_angle:
        planner_label = "any_angle"
    elif is_hybrid:
        planner_label = "hybrid_astar"
    else:
        planner_label = planner_key
    if is_astar:
        cells = _run_astar(
            geo=geo,
            blocked=blocked,
            caution=caution,
            ais_norm=ais_norm,
            start=(sr, sc),
            goal=(gr, gc),
            km_per_row=km_per_row,
            km_per_col_min=km_per_col_min,
            caution_penalty=caution_penalty,
            corridor_reward=corridor_reward,
            near_blocked=near_blocked,
            near_blocked_penalty=near_blocked_penalty,
            uncertainty_penalty=unc_penalty,
            risk_penalty=risk_penalty,
        )
    elif is_dstar:
        cells = _run_dstar_lite_static(
            geo=geo,
            blocked=blocked,
            caution=caution,
            ais_norm=ais_norm,
            start=(sr, sc),
            goal=(gr, gc),
            caution_penalty=caution_penalty,
            corridor_reward=corridor_reward,
            near_blocked=near_blocked,
            near_blocked_penalty=near_blocked_penalty,
            uncertainty_penalty=unc_penalty,
            risk_penalty=risk_penalty,
        )
    elif is_any_angle:
        cells = _run_theta_star(
            geo=geo,
            blocked=blocked,
            caution=caution,
            ais_norm=ais_norm,
            start=(sr, sc),
            goal=(gr, gc),
            km_per_row=km_per_row,
            km_per_col_min=km_per_col_min,
            caution_penalty=caution_penalty,
            corridor_reward=corridor_reward,
            near_blocked=near_blocked,
            near_blocked_penalty=near_blocked_penalty,
            uncertainty_penalty=unc_penalty,
            risk_penalty=risk_penalty,
        )
    elif is_hybrid:
        cells = _run_hybrid_astar(
            geo=geo,
            blocked=blocked,
            caution=caution,
            ais_norm=ais_norm,
            start=(sr, sc),
            goal=(gr, gc),
            km_per_row=km_per_row,
            km_per_col_min=km_per_col_min,
            caution_penalty=caution_penalty,
            corridor_reward=corridor_reward,
            near_blocked=near_blocked,
            near_blocked_penalty=near_blocked_penalty,
            uncertainty_penalty=unc_penalty,
            risk_penalty=risk_penalty,
        )
    else:
        raise PlanningError(
            "Unsupported planner={planner}, expected one of astar, dstar_lite, "
            "dstar_lite_recompute, any_angle, hybrid_astar".format(planner=planner)
        )

    raw_cells = cells[:]
    max_turn_limit_deg = 110.0
    smoothed_fallback = False
    smoothed_fallback_reason = ""
    if smoothing:
        if is_any_angle or is_hybrid:
            cells = _smooth_cells_los_constrained(raw_cells, blocked, max_turn_deg=max_turn_limit_deg)
        else:
            cells = _smooth_cells_los(raw_cells, blocked)
    if not _is_path_feasible(cells, blocked):
        cells = raw_cells[:]
        smoothed_fallback = True
        smoothed_fallback_reason = "smoothed_path_crosses_blocked"

    coords = _cells_to_coords(geo, cells)
    raw_coords = _cells_to_coords(geo, raw_cells)
    metrics = _collect_path_metrics(
        cells=cells,
        geo=geo,
        caution=caution,
        ais_norm=ais_norm,
        near_blocked=near_blocked,
        caution_penalty=caution_penalty,
        corridor_reward=corridor_reward,
        uncertainty_penalty=unc_penalty,
        risk_penalty=risk_penalty,
    )
    raw_distance_km = float(metrics["distance_km"])
    base_distance_km = float(metrics["base_distance_km"])
    caution_len_km = float(metrics["caution_len_km"])
    cost_caution_extra_km = float(metrics["cost_caution_extra_km"])
    cost_corridor_discount_km = float(metrics["cost_corridor_discount_km"])
    cost_uncertainty_extra_km = float(metrics["cost_uncertainty_extra_km"])
    cost_risk_extra_km = float(metrics["cost_risk_extra_km"])
    corridor_vals = list(metrics["corridor_vals"])
    uncertainty_penalty_vals = list(metrics["uncertainty_penalty_vals"])
    risk_penalty_vals = list(metrics["risk_penalty_vals"])
    caution_hits = int(metrics["caution_hits"])
    near_blocked_hits = int(metrics["near_blocked_hits"])
    sample_count = int(metrics["sample_count"])

    s_lat0, s_lon0 = geo.rc_to_latlon(*s_rc)
    s_lat1, s_lon1 = geo.rc_to_latlon(*s_rc_adj)
    g_lat0, g_lon0 = geo.rc_to_latlon(*g_rc)
    g_lat1, g_lon1 = geo.rc_to_latlon(*g_rc_adj)
    start_adjust_km = haversine_km(s_lat0, s_lon0, s_lat1, s_lon1)
    goal_adjust_km = haversine_km(g_lat0, g_lon0, g_lat1, g_lon1)

    explain = {
        "distance_km": round(float(raw_distance_km), 3),
        "distance_nm": round(float(raw_distance_km) * 0.539957, 3),
        "caution_len_km": round(float(caution_len_km), 3),
        "corridor_alignment": round(float(np.mean(corridor_vals) if corridor_vals else 0.0), 3),
        "corridor_alignment_p50": round(float(np.percentile(corridor_vals, 50)) if corridor_vals else 0.0, 3),
        "corridor_alignment_p90": round(float(np.percentile(corridor_vals, 90)) if corridor_vals else 0.0, 3),
        "caution_mode": caution_mode,
        "effective_caution_penalty": round(float(caution_penalty), 4),
        "effective_corridor_reward": round(float(corridor_reward), 4),
        "effective_near_blocked_penalty": round(float(near_blocked_penalty), 4),
        "route_cost_base_km": round(float(base_distance_km), 3),
        "route_cost_caution_extra_km": round(float(cost_caution_extra_km), 3),
        "route_cost_corridor_discount_km": round(float(cost_corridor_discount_km), 3),
        "route_cost_uncertainty_extra_km": round(float(cost_uncertainty_extra_km), 3),
        "route_cost_risk_extra_km": round(float(cost_risk_extra_km), 3),
        "route_cost_effective_km": round(
            float(
                base_distance_km
                + cost_caution_extra_km
                + cost_uncertainty_extra_km
                + cost_risk_extra_km
                - cost_corridor_discount_km
            ),
            3,
        ),
        "risk_mode": str(risk_meta.get("risk_mode", risk_mode)),
        "risk_layer": str(risk_meta.get("risk_layer", "risk_mean")),
        "risk_lambda": round(float(risk_meta.get("risk_lambda", 0.0)), 4),
        "risk_penalty_mean": round(float(np.mean(risk_penalty_vals) if risk_penalty_vals else 0.0), 4),
        "risk_penalty_p90": round(float(np.percentile(risk_penalty_vals, 90)) if risk_penalty_vals else 0.0, 4),
        "risk_meta": risk_meta,
        "uncertainty_uplift_enabled": bool(uncertainty_uplift),
        "uncertainty_uplift_scale": round(float(unc_meta.get("uplift_scale", 0.0)), 4),
        "uncertainty_threshold": round(float(unc_meta.get("threshold", 0.0)), 4),
        "uncertainty_temperature": round(float(unc_meta.get("temperature", 1.0)), 4),
        "uncertainty_profile_available": bool(unc_meta.get("profile_available", False)),
        "uncertainty_penalty_mean": round(float(np.mean(uncertainty_penalty_vals) if uncertainty_penalty_vals else 0.0), 4),
        "uncertainty_penalty_p90": round(
            float(np.percentile(uncertainty_penalty_vals, 90)) if uncertainty_penalty_vals else 0.0,
            4,
        ),
        "uncertainty_mean": round(float(unc_meta.get("uncertainty_mean", 0.0)), 4),
        "uncertainty_p90": round(float(unc_meta.get("uncertainty_p90", 0.0)), 4),
        "uncertainty_calibration": unc_meta,
        "smoothing": bool(smoothing),
        "smoothing_feasible": bool(not smoothed_fallback),
        "smoothing_fallback_reason": smoothed_fallback_reason,
        "turn_constraint_max_deg": float(max_turn_limit_deg) if (is_any_angle or is_hybrid) else None,
        "raw_max_turn_deg": round(float(_max_turn_angle_deg(raw_cells)), 3),
        "smoothed_max_turn_deg": round(float(_max_turn_angle_deg(cells)), 3),
        "raw_points": len(raw_cells),
        "smoothed_points": len(cells),
        "start_adjusted": s_rc_adj != s_rc,
        "goal_adjusted": g_rc_adj != g_rc,
        "start_adjust_km": round(float(start_adjust_km), 3),
        "goal_adjust_km": round(float(goal_adjust_km), 3),
        "blocked_ratio": round(float(blocked.mean()), 4),
        "adjacent_blocked_ratio": round(float(near_blocked_hits / max(1, sample_count)), 4),
        "caution_cell_ratio": round(float(caution_hits / max(1, sample_count)), 4),
        "planner": planner_label,
        "grid_bounds": {
            "lat_min": bounds.lat_min,
            "lat_max": bounds.lat_max,
            "lon_min": bounds.lon_min,
            "lon_max": bounds.lon_max,
        },
    }

    risk_lambda = float(risk_meta.get("risk_lambda", 0.0))
    raw_risk_values = _risk_raw_values_from_penalty_values(risk_penalty_vals, risk_lambda=risk_lambda)
    constraint = _evaluate_risk_constraint(
        mode=risk_constraint_mode,
        budget=float(risk_budget),
        confidence_level=float(confidence_level),
        raw_risk_values=raw_risk_values,
    )
    explain.update(
        {
            "risk_constraint_mode": constraint["mode"],
            "risk_budget": round(float(constraint["budget"]), 4),
            "risk_confidence_level": round(float(constraint["confidence_level"]), 4),
            "risk_constraint_metric_name": str(constraint["metric_name"]),
            "risk_constraint_metric": round(float(constraint["metric"]), 6),
            "risk_budget_usage": round(float(constraint["usage"]), 6),
            "risk_constraint_satisfied": bool(constraint["satisfied"]),
            "risk_constraint_sample_count": int(constraint["sample_count"]),
            "risk_constraint_tail_count": int(constraint["tail_count"]),
        }
    )
    if constraint["mode"] != "none" and not bool(constraint["satisfied"]):
        raise PlanningError(
            "Risk constraint violated: "
            f"{constraint['metric_name']}={constraint['metric']:.4f} > budget={constraint['budget']:.4f}"
        )

    if is_any_angle or is_hybrid:
        display_coords = coords
    else:
        display_coords = _build_display_coordinates(coords)
    route_geojson = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {
            **explain,
            "display_coordinates": display_coords,
            "raw_coordinates": raw_coords,
            "feasible_smoothed_coordinates": coords,
        },
    }
    return PlanResult(route_geojson=route_geojson, explain=explain)
