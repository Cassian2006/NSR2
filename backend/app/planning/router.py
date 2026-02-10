from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.model.infer import run_unet_inference


EARTH_RADIUS_KM = 6371.0088


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


def _latlon_to_rc(lat: float, lon: float, h: int, w: int, bounds: GridBounds) -> tuple[int, int]:
    if not (bounds.lat_min <= lat <= bounds.lat_max and bounds.lon_min <= lon <= bounds.lon_max):
        raise PlanningError(
            f"Point out of grid bounds: lat={lat}, lon={lon}, "
            f"bounds=({bounds.lat_min},{bounds.lat_max},{bounds.lon_min},{bounds.lon_max})"
        )
    r = int(round((bounds.lat_max - lat) / (bounds.lat_max - bounds.lat_min) * (h - 1)))
    c = int(round((lon - bounds.lon_min) / (bounds.lon_max - bounds.lon_min) * (w - 1)))
    r = min(max(r, 0), h - 1)
    c = min(max(c, 0), w - 1)
    return r, c


def _rc_to_latlon(r: int, c: int, h: int, w: int, bounds: GridBounds) -> tuple[float, float]:
    lat = bounds.lat_max - (r / max(1, h - 1)) * (bounds.lat_max - bounds.lat_min)
    lon = bounds.lon_min + (c / max(1, w - 1)) * (bounds.lon_max - bounds.lon_min)
    return float(lat), float(lon)


def _neighbors(r: int, c: int, h: int, w: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


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


def _smooth_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(cells) <= 2:
        return cells
    out = [cells[0]]
    prev_dr = None
    prev_dc = None
    for i in range(1, len(cells)):
        r0, c0 = cells[i - 1]
        r1, c1 = cells[i]
        dr = r1 - r0
        dc = c1 - c0
        if prev_dr is None:
            prev_dr, prev_dc = dr, dc
            out.append(cells[i])
            continue
        if dr == prev_dr and dc == prev_dc:
            out[-1] = cells[i]
        else:
            out.append(cells[i])
        prev_dr, prev_dc = dr, dc
    return out


def _load_ais_heatmap(settings: Settings, timestamp: str, shape: tuple[int, int]) -> np.ndarray:
    root = settings.ais_heatmap_root
    if root.exists():
        hits = list(root.rglob(f"{timestamp}.npy"))
        if hits:
            arr = np.load(hits[0]).astype(np.float32)
            if arr.shape == shape:
                return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.zeros(shape, dtype=np.float32)


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
) -> PlanResult:
    ann_dir = settings.annotation_pack_root / timestamp
    blocked_path = ann_dir / "blocked_mask.npy"
    if not blocked_path.exists():
        raise PlanningError(f"blocked_mask missing for timestamp={timestamp}: {blocked_path}")
    blocked_bathy = np.load(blocked_path).astype(np.uint8) > 0
    h, w = blocked_bathy.shape

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
    free_cells = np.argwhere(~blocked)
    if free_cells.size == 0:
        raise PlanningError("No navigable cells available after blocked-mask fusion.")
    caution = unet_pred == 1
    ais = _load_ais_heatmap(settings, timestamp=timestamp, shape=(h, w))
    ais = np.clip(ais, 0.0, None)
    ais_max = float(ais.max())
    ais_norm = ais / ais_max if ais_max > 1e-8 else np.zeros_like(ais)

    bounds = GridBounds(
        lat_min=60.0,
        lat_max=86.0,
        lon_min=-180.0,
        lon_max=180.0,
    )
    s_rc = _latlon_to_rc(start[0], start[1], h, w, bounds)
    g_rc = _latlon_to_rc(goal[0], goal[1], h, w, bounds)
    s_rc_adj = _nearest_unblocked(s_rc, blocked, free_cells)
    g_rc_adj = _nearest_unblocked(g_rc, blocked, free_cells)

    lat_step = (bounds.lat_max - bounds.lat_min) / max(1, h - 1)
    lon_step = (bounds.lon_max - bounds.lon_min) / max(1, w - 1)
    km_per_row = 111.32 * lat_step
    km_per_col_min = 111.32 * max(0.05, math.cos(math.radians(bounds.lat_max))) * lon_step
    caution_penalty = 0.12 if caution_mode == "tie_breaker" else 0.0
    corridor_reward = max(0.0, float(corridor_bias)) * 0.15

    gscore = np.full((h, w), np.inf, dtype=np.float64)
    closed = np.zeros((h, w), dtype=bool)
    came_from: dict[tuple[int, int], tuple[int, int]] = {}

    sr, sc = s_rc_adj
    gr, gc = g_rc_adj
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

        lat0, lon0 = _rc_to_latlon(r, c, h, w, bounds)
        for rr, cc in _neighbors(r, c, h, w):
            if closed[rr, cc] or blocked[rr, cc]:
                continue
            lat1, lon1 = _rc_to_latlon(rr, cc, h, w, bounds)
            step_km = haversine_km(lat0, lon0, lat1, lon1)
            mult = 1.0
            if caution[rr, cc]:
                mult += caution_penalty
            mult -= corridor_reward * float(ais_norm[rr, cc])
            mult = max(0.15, mult)
            cand = cur_g + step_km * mult
            if cand < gscore[rr, cc]:
                gscore[rr, cc] = cand
                came_from[(rr, cc)] = (r, c)
                hval = _heuristic_km(rr, cc, gr, gc, km_per_row=km_per_row, km_per_col_min=km_per_col_min)
                heapq.heappush(heap, (cand + hval, cand, rr, cc))

    if not np.isfinite(gscore[gr, gc]):
        raise PlanningError("No feasible route found under current blocked constraints.")

    cells = _reconstruct_path(came_from, (gr, gc))
    raw_cells = cells[:]
    if smoothing:
        cells = _smooth_cells(cells)

    coords: list[list[float]] = []
    raw_distance_km = 0.0
    caution_len_km = 0.0
    corridor_vals: list[float] = []
    prev_lat = None
    prev_lon = None
    for idx, (r, c) in enumerate(cells):
        lat, lon = _rc_to_latlon(r, c, h, w, bounds)
        coords.append([lon, lat])
        corridor_vals.append(float(ais_norm[r, c]))
        if idx > 0 and prev_lat is not None and prev_lon is not None:
            step = haversine_km(prev_lat, prev_lon, lat, lon)
            raw_distance_km += step
            if caution[r, c]:
                caution_len_km += step
        prev_lat, prev_lon = lat, lon

    explain = {
        "distance_km": round(float(raw_distance_km), 3),
        "distance_nm": round(float(raw_distance_km) * 0.539957, 3),
        "caution_len_km": round(float(caution_len_km), 3),
        "corridor_alignment": round(float(np.mean(corridor_vals) if corridor_vals else 0.0), 3),
        "caution_mode": caution_mode,
        "smoothing": bool(smoothing),
        "raw_points": len(raw_cells),
        "smoothed_points": len(cells),
        "start_adjusted": s_rc_adj != s_rc,
        "goal_adjusted": g_rc_adj != g_rc,
        "blocked_ratio": round(float(blocked.mean()), 4),
    }

    route_geojson = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": explain,
    }
    return PlanResult(route_geojson=route_geojson, explain=explain)
