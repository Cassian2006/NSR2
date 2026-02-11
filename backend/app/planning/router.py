from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.core.geo import load_grid_geo
from app.model.infer import run_unet_inference


EARTH_RADIUS_KM = 6371.0088
DSTAR_INCREMENTAL_CHANGED_RATIO = 0.08
DSTAR_INCREMENTAL_CHANGED_MIN_CELLS = 64


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
) -> tuple[float, float, float]:
    """Return (caution_ratio, corridor_mean, near_blocked_ratio) along the segment."""
    h, w = caution.shape
    traced = _trace_line_cells(from_rc, to_rc, (h, w))
    sampled = traced[1:] if len(traced) > 1 else traced
    if not sampled:
        return 0.0, 0.0, 0.0

    caution_hits = 0
    near_hits = 0
    corridor_vals: list[float] = []
    for rr, rc in sampled:
        caution_hits += int(bool(caution[rr, rc]))
        if near_blocked is not None:
            near_hits += int(bool(near_blocked[rr, rc]))
        corridor_vals.append(float(ais_norm[rr, rc]))

    n = len(sampled)
    caution_ratio = float(caution_hits / max(1, n))
    near_ratio = float(near_hits / max(1, n))
    corridor_mean = float(np.mean(corridor_vals) if corridor_vals else 0.0)
    return caution_ratio, corridor_mean, near_ratio


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
) -> float:
    fr, fc = from_rc
    tr, tc = to_rc
    lat0, lon0 = geo.rc_to_latlon(fr, fc)
    lat1, lon1 = geo.rc_to_latlon(tr, tc)
    step_km = haversine_km(lat0, lon0, lat1, lon1)
    caution_ratio, corridor_mean, near_ratio = _segment_cell_stats(
        from_rc=from_rc,
        to_rc=to_rc,
        caution=caution,
        ais_norm=ais_norm,
        near_blocked=near_blocked,
    )
    mult = 1.0 + caution_penalty * caution_ratio + near_blocked_penalty * near_ratio
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
) -> dict:
    if len(cells) < 2:
        return {
            "distance_km": 0.0,
            "base_distance_km": 0.0,
            "caution_len_km": 0.0,
            "cost_caution_extra_km": 0.0,
            "cost_corridor_discount_km": 0.0,
            "caution_hits": 0,
            "near_blocked_hits": 0,
            "sample_count": 0,
            "corridor_vals": [],
        }

    h, w = caution.shape
    total_distance_km = 0.0
    caution_len_km = 0.0
    cost_caution_extra_km = 0.0
    cost_corridor_discount_km = 0.0
    caution_hits = 0
    near_blocked_hits = 0
    sample_count = 0
    corridor_vals: list[float] = []

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
        for rr, rc in sampled:
            caution_seg_hits += int(bool(caution[rr, rc]))
            near_blocked_seg_hits += int(bool(near_blocked[rr, rc]))
            corridor_seg_vals.append(float(ais_norm[rr, rc]))

        seg_samples = len(sampled)
        sample_count += seg_samples
        caution_hits += caution_seg_hits
        near_blocked_hits += near_blocked_seg_hits
        corridor_vals.extend(corridor_seg_vals)

        caution_ratio = float(caution_seg_hits / max(1, seg_samples))
        corridor_mean = float(np.mean(corridor_seg_vals) if corridor_seg_vals else 0.0)

        caution_len_km += segment_km * caution_ratio
        cost_caution_extra_km += segment_km * caution_penalty * caution_ratio
        cost_corridor_discount_km += segment_km * corridor_reward * corridor_mean

    return {
        "distance_km": total_distance_km,
        "base_distance_km": total_distance_km,
        "caution_len_km": caution_len_km,
        "cost_caution_extra_km": cost_caution_extra_km,
        "cost_corridor_discount_km": cost_corridor_discount_km,
        "caution_hits": caution_hits,
        "near_blocked_hits": near_blocked_hits,
        "sample_count": sample_count,
        "corridor_vals": corridor_vals,
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
            )

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
                )
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
    ) -> None:
        self.geo = geo
        self.blocked = blocked.copy()
        self.caution = caution
        self.ais_norm = ais_norm
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
    ) -> None:
        if new_blocked.shape != (self.h, self.w):
            raise PlanningError("Dynamic update shape mismatch in D* Lite planner.")
        if changed_cells.size == 0:
            return
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
) -> PlanResult:
    if len(timestamps) < 2:
        raise PlanningError("Dynamic replanning requires at least 2 timestamps.")
    if advance_steps < 1:
        raise PlanningError("advance_steps must be >= 1")
    valid_caution_modes = {"tie_breaker", "budget", "minimize", "strict"}
    if caution_mode not in valid_caution_modes:
        raise PlanningError(f"Unsupported caution_mode={caution_mode}, expected one of {sorted(valid_caution_modes)}")

    states = [
        _load_grid_state(
            settings=settings,
            timestamp=ts,
            model_version=model_version,
            blocked_sources=blocked_sources,
            caution_mode=caution_mode,
        )
        for ts in timestamps
    ]
    h, w = states[0].blocked.shape
    for state in states[1:]:
        if state.blocked.shape != (h, w):
            raise PlanningError("Dynamic replanning requires aligned grid shape across timestamps.")

    planner_key = planner.strip().lower()
    is_astar = planner_key in {"astar", "a_star"}
    is_dstar = planner_key in {"dstar_lite", "dstar-lite", "dstar"}
    is_any_angle = planner_key in {"any_angle", "any-angle", "theta", "theta_star", "thetastar"}
    is_hybrid = planner_key in {"hybrid_astar", "hybrid-a*", "hybrid", "hybrid_a_star"}
    if is_astar:
        planner_label = "astar"
    elif is_dstar:
        planner_label = "dstar_lite"
    elif is_any_angle:
        planner_label = "any_angle"
    elif is_hybrid:
        planner_label = "hybrid_astar"
    else:
        planner_label = planner_key
    if not (is_astar or is_dstar or is_any_angle or is_hybrid):
        raise PlanningError(
            f"Unsupported planner={planner}, expected one of astar, dstar_lite, any_angle, hybrid_astar"
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
    executed_points = 0
    executed_caution_points = 0
    executed_adjacent_blocked_points = 0
    corridor_vals: list[float] = []
    total_distance_km = 0.0
    base_distance_km = 0.0
    caution_len_km = 0.0
    cost_caution_extra_km = 0.0
    cost_corridor_discount_km = 0.0
    replans: list[dict] = []
    route_cells_executed = 0
    dynamic_notes: list[str] = []

    current_start = start_rc
    current_goal = goal_rc

    dstar: _DStarLiteIncremental | None = None
    current_cells: list[tuple[int, int]] = []

    for step_idx, state in enumerate(states):
        step_t0 = time.perf_counter()
        step_update_mode = "init" if step_idx == 0 else "none"
        step_changed = 0
        goal_now = _nearest_unblocked(current_goal, state.blocked, state.free_cells)
        if goal_now != current_goal:
            current_goal = goal_now
            dstar = None
            dynamic_notes.append(f"goal_adjusted_at_step_{step_idx}")

        if is_astar:
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
            )
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
            )
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
            )
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
                )
                step_update_mode = "rebuild"
            else:
                diff_mask = dstar.blocked != state.blocked
                step_changed = int(np.count_nonzero(diff_mask))
                change_ratio = float(step_changed) / float(h * w)
                incremental_limit = max(DSTAR_INCREMENTAL_CHANGED_MIN_CELLS, int(h * w * DSTAR_INCREMENTAL_CHANGED_RATIO))
                if step_changed and step_changed <= incremental_limit:
                    changed = np.argwhere(diff_mask)
                    dstar.update_blocked(state.blocked, changed, auto_compute=False)
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
                    )
                    step_update_mode = "rebuild"
                    dynamic_notes.append(
                        f"large_update_rebuild_step_{step_idx}: changed={step_changed} ratio={change_ratio:.4f}"
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
                    )
                    step_update_mode = "rebuild"
                if dstar.start != current_start:
                    dstar.move_start(current_start, auto_compute=False)
                dstar.sync_if_needed()
            current_cells = dstar.extract_path()

        raw_points = len(current_cells)
        move_cells = current_cells
        smooth_cells = _smooth_cells_los(current_cells, state.blocked) if smoothing else current_cells
        smoothed_points = len(smooth_cells)
        if len(move_cells) < 2:
            break

        move_edges = min(advance_steps, len(move_cells) - 1) if step_idx < len(states) - 1 else len(move_cells) - 1
        step_distance_km = 0.0
        step_caution_km = 0.0
        step_caution_extra_km = 0.0
        step_corridor_discount_km = 0.0
        step_samples = 0
        step_caution_hits = 0
        step_near_blocked_hits = 0
        metric_cells = move_cells[: move_edges + 1]
        near_blocked = state.near_blocked if state.near_blocked is not None else _build_near_blocked_mask(state.blocked)
        step_metrics = _collect_path_metrics(
            cells=metric_cells,
            geo=state.geo,
            caution=state.caution,
            ais_norm=state.ais_norm,
            near_blocked=near_blocked,
            caution_penalty=caution_penalty,
            corridor_reward=corridor_reward,
        )
        step_distance_km = float(step_metrics["distance_km"])
        step_caution_km = float(step_metrics["caution_len_km"])
        step_caution_extra_km = float(step_metrics["cost_caution_extra_km"])
        step_corridor_discount_km = float(step_metrics["cost_corridor_discount_km"])
        step_samples = int(step_metrics["sample_count"])
        step_caution_hits = int(step_metrics["caution_hits"])
        step_near_blocked_hits = int(step_metrics["near_blocked_hits"])
        corridor_vals.extend(step_metrics["corridor_vals"])
        for idx in range(1, move_edges + 1):
            pr, pc = move_cells[idx - 1]
            cr, cc = move_cells[idx]
            plat, plon = state.geo.rc_to_latlon(pr, pc)
            clat, clon = state.geo.rc_to_latlon(cr, cc)
            if not executed_coords:
                executed_coords.append([plon, plat])
            executed_coords.append([clon, clat])

        executed_points += step_samples
        executed_caution_points += step_caution_hits
        executed_adjacent_blocked_points += step_near_blocked_hits

        total_distance_km += step_distance_km
        base_distance_km += step_distance_km
        caution_len_km += step_caution_km
        cost_caution_extra_km += step_caution_extra_km
        cost_corridor_discount_km += step_corridor_discount_km
        route_cells_executed += move_edges

        moved_to = move_cells[move_edges]
        current_start = _nearest_unblocked(moved_to, state.blocked, state.free_cells)
        step_runtime_ms = (time.perf_counter() - step_t0) * 1000.0
        if step_idx > 0 and step_changed == 0:
            step_changed = int(np.count_nonzero(states[step_idx - 1].blocked != state.blocked))
        replans.append(
            {
                "step": step_idx,
                "timestamp": state.timestamp,
                "runtime_ms": round(float(step_runtime_ms), 3),
                "raw_points": raw_points,
                "smoothed_points": smoothed_points,
                "moved_edges": move_edges,
                "moved_distance_km": round(float(step_distance_km), 3),
                "changed_blocked_cells": step_changed if step_idx > 0 else 0,
                "update_mode": step_update_mode,
            }
        )

        if current_start == current_goal:
            break

    incremental_steps = int(sum(1 for r in replans if r.get("update_mode") == "incremental"))
    rebuild_steps = int(sum(1 for r in replans if r.get("update_mode") == "rebuild"))
    if is_dstar:
        planner_label = "dstar_lite_incremental"
    elif is_any_angle:
        planner_label = "any_angle_recompute"
    elif is_hybrid:
        planner_label = "hybrid_astar_recompute"
    else:
        planner_label = "astar_recompute"
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
        "route_cost_effective_km": round(float(base_distance_km + cost_caution_extra_km - cost_corridor_discount_km), 3),
        "smoothing": bool(smoothing),
        "start_adjusted": start_adjusted,
        "goal_adjusted": goal_adjusted,
        "start_adjust_km": round(float(start_adjust_km), 3),
        "goal_adjust_km": round(float(goal_adjust_km), 3),
        "planner": planner_label,
        "dynamic_replans": replans,
        "dynamic_advance_steps": int(advance_steps),
        "dynamic_timestamps": timestamps,
        "dynamic_notes": dynamic_notes,
        "dynamic_incremental_steps": incremental_steps,
        "dynamic_rebuild_steps": rebuild_steps,
        "dynamic_incremental_threshold_ratio": round(float(DSTAR_INCREMENTAL_CHANGED_RATIO), 4),
        "executed_edges": int(route_cells_executed),
        "blocked_ratio_last": round(float(states[-1].blocked.mean()), 4),
        "adjacent_blocked_ratio": round(float(executed_adjacent_blocked_points / max(1, executed_points)), 4),
        "caution_cell_ratio": round(float(executed_caution_points / max(1, executed_points)), 4),
        "replan_runtime_ms_total": round(float(sum(float(r.get("runtime_ms", 0.0)) for r in replans)), 3),
        "replan_runtime_ms_mean": round(
            float(sum(float(r.get("runtime_ms", 0.0)) for r in replans) / max(1, len(replans))),
            3,
        ),
        "replan_changed_cells_total": int(sum(int(r.get("changed_blocked_cells", 0)) for r in replans)),
    }

    if is_any_angle or is_hybrid:
        display_coords = executed_coords
    else:
        display_coords = _build_display_coordinates(executed_coords)
    route_geojson = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": executed_coords},
        "properties": {**explain, "display_coordinates": display_coords},
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
) -> PlanResult:
    valid_caution_modes = {"tie_breaker", "budget", "minimize", "strict"}
    if caution_mode not in valid_caution_modes:
        raise PlanningError(f"Unsupported caution_mode={caution_mode}, expected one of {sorted(valid_caution_modes)}")

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
    is_dstar = planner_key in {"dstar_lite", "dstar-lite", "dstar"}
    is_any_angle = planner_key in {"any_angle", "any-angle", "theta", "theta_star", "thetastar"}
    is_hybrid = planner_key in {"hybrid_astar", "hybrid-a*", "hybrid", "hybrid_a_star"}
    if is_astar:
        planner_label = "astar"
    elif is_dstar:
        planner_label = "dstar_lite"
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
        )
    else:
        raise PlanningError(
            f"Unsupported planner={planner}, expected one of astar, dstar_lite, any_angle, hybrid_astar"
        )

    raw_cells = cells[:]
    if smoothing:
        cells = _smooth_cells_los(cells, blocked)
    if is_any_angle or is_hybrid:
        cells = _expand_cells_supercover(cells, blocked.shape)

    coords: list[list[float]] = []
    for r, c in cells:
        lat, lon = geo.rc_to_latlon(r, c)
        coords.append([lon, lat])
    metrics = _collect_path_metrics(
        cells=cells,
        geo=geo,
        caution=caution,
        ais_norm=ais_norm,
        near_blocked=near_blocked,
        caution_penalty=caution_penalty,
        corridor_reward=corridor_reward,
    )
    raw_distance_km = float(metrics["distance_km"])
    base_distance_km = float(metrics["base_distance_km"])
    caution_len_km = float(metrics["caution_len_km"])
    cost_caution_extra_km = float(metrics["cost_caution_extra_km"])
    cost_corridor_discount_km = float(metrics["cost_corridor_discount_km"])
    corridor_vals = list(metrics["corridor_vals"])
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
        "route_cost_effective_km": round(float(base_distance_km + cost_caution_extra_km - cost_corridor_discount_km), 3),
        "smoothing": bool(smoothing),
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

    if is_any_angle or is_hybrid:
        display_coords = coords
    else:
        display_coords = _build_display_coordinates(coords)
    route_geojson = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {**explain, "display_coordinates": display_coords},
    }
    return PlanResult(route_geojson=route_geojson, explain=explain)
