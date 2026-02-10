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
    """Integer Bresenham LOS check; false when a blocked cell is crossed."""
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    h, w = blocked.shape
    while True:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
        if r == r1 and c == c1:
            return True
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


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


def _load_ais_heatmap(settings: Settings, timestamp: str, shape: tuple[int, int]) -> np.ndarray:
    root = settings.ais_heatmap_root
    if root.exists():
        hits = list(root.rglob(f"{timestamp}.npy"))
        if hits:
            arr = np.load(hits[0]).astype(np.float32)
            if arr.shape == shape:
                return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.zeros(shape, dtype=np.float32)


def _policy_scalars(caution_mode: str, corridor_bias: float) -> tuple[float, float]:
    if caution_mode == "tie_breaker":
        caution_penalty = 0.12
        corridor_reward_scale = 0.15
    elif caution_mode == "budget":
        caution_penalty = 0.24
        corridor_reward_scale = 0.08
    elif caution_mode == "minimize":
        caution_penalty = 0.45
        corridor_reward_scale = 0.04
    else:  # strict
        caution_penalty = 0.0
        corridor_reward_scale = 0.0
    corridor_reward = max(0.0, float(corridor_bias)) * corridor_reward_scale
    return caution_penalty, corridor_reward


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
) -> float:
    fr, fc = from_rc
    tr, tc = to_rc
    lat0, lon0 = geo.rc_to_latlon(fr, fc)
    lat1, lon1 = geo.rc_to_latlon(tr, tc)
    step_km = haversine_km(lat0, lon0, lat1, lon1)
    mult = 1.0
    if caution[tr, tc]:
        mult += caution_penalty
    mult -= corridor_reward * float(ais_norm[tr, tc])
    mult = max(0.15, mult)
    return step_km * mult


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
            )
            if cand < gscore[rr, cc]:
                gscore[rr, cc] = cand
                came_from[(rr, cc)] = (r, c)
                hval = _heuristic_km(rr, cc, gr, gc, km_per_row=km_per_row, km_per_col_min=km_per_col_min)
                heapq.heappush(heap, (cand + hval, cand, rr, cc))

    if not np.isfinite(gscore[gr, gc]):
        raise PlanningError("No feasible route found under current blocked constraints.")
    return _reconstruct_path(came_from, (gr, gc))


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
        )
        self.g = g_to_goal.copy()
        self.rhs = g_to_goal.copy()

        self.open_heap: list[tuple[float, float, int, int]] = []
        self.open_best: dict[tuple[int, int], tuple[float, float]] = {}

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

    def update_blocked(self, new_blocked: np.ndarray, changed_cells: np.ndarray) -> None:
        if new_blocked.shape != (self.h, self.w):
            raise PlanningError("Dynamic update shape mismatch in D* Lite planner.")
        self.blocked = new_blocked.copy()
        touched: set[tuple[int, int]] = set()
        for rc in changed_cells:
            r = int(rc[0])
            c = int(rc[1])
            touched.add((r, c))
            for pr, pc in _neighbors(r, c, self.h, self.w):
                touched.add((pr, pc))
        for u in touched:
            self._update_vertex(u)
        self.compute_shortest_path()

    def move_start(self, new_start: tuple[int, int]) -> None:
        if new_start == self.start:
            return
        self.km += self._heuristic(self.last, new_start)
        self.last = new_start
        self.start = new_start
        self.compute_shortest_path()

    def extract_path(self, max_steps: int | None = None) -> list[tuple[int, int]]:
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
    if planner_key not in {"astar", "a_star", "dstar_lite", "dstar-lite", "dstar"}:
        raise PlanningError(f"Unsupported planner={planner}, expected astar or dstar_lite")

    caution_penalty, corridor_reward = _policy_scalars(caution_mode, corridor_bias)
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

    executed_coords: list[list[float]] = []
    corridor_vals: list[float] = []
    total_distance_km = 0.0
    caution_len_km = 0.0
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

        if planner_key in {"astar", "a_star"}:
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
                )
                step_update_mode = "rebuild"
            else:
                changed = np.argwhere(dstar.blocked != state.blocked)
                step_changed = int(changed.shape[0]) if changed.ndim == 2 else 0
                change_ratio = float(step_changed) / float(h * w)
                if step_changed and change_ratio <= 0.02:
                    dstar.update_blocked(state.blocked, changed)
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
                    )
                    step_update_mode = "rebuild"
                if dstar.start != current_start:
                    dstar.move_start(current_start)
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
        for idx in range(1, move_edges + 1):
            pr, pc = move_cells[idx - 1]
            cr, cc = move_cells[idx]
            plat, plon = state.geo.rc_to_latlon(pr, pc)
            clat, clon = state.geo.rc_to_latlon(cr, cc)
            step = haversine_km(plat, plon, clat, clon)
            step_distance_km += step
            if state.caution[cr, cc]:
                step_caution_km += step
            corridor_vals.append(float(state.ais_norm[cr, cc]))
            if not executed_coords:
                executed_coords.append([plon, plat])
            executed_coords.append([clon, clat])

        total_distance_km += step_distance_km
        caution_len_km += step_caution_km
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

    explain = {
        "distance_km": round(float(total_distance_km), 3),
        "distance_nm": round(float(total_distance_km) * 0.539957, 3),
        "caution_len_km": round(float(caution_len_km), 3),
        "corridor_alignment": round(float(np.mean(corridor_vals) if corridor_vals else 0.0), 3),
        "caution_mode": caution_mode,
        "effective_caution_penalty": round(float(caution_penalty), 4),
        "effective_corridor_reward": round(float(corridor_reward), 4),
        "smoothing": bool(smoothing),
        "start_adjusted": start_adjusted,
        "goal_adjusted": goal_adjusted,
        "planner": "dstar_lite_incremental" if planner_key in {"dstar_lite", "dstar-lite", "dstar"} else "astar_recompute",
        "dynamic_replans": replans,
        "dynamic_advance_steps": int(advance_steps),
        "dynamic_timestamps": timestamps,
        "dynamic_notes": dynamic_notes,
        "executed_edges": int(route_cells_executed),
        "blocked_ratio_last": round(float(states[-1].blocked.mean()), 4),
    }

    route_geojson = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": executed_coords},
        "properties": explain,
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

    lat_step = (bounds.lat_max - bounds.lat_min) / max(1, h - 1)
    lon_step = (bounds.lon_max - bounds.lon_min) / max(1, w - 1)
    km_per_row = 111.32 * lat_step
    km_per_col_min = 111.32 * max(0.05, math.cos(math.radians(bounds.lat_max))) * lon_step
    if caution_mode == "tie_breaker":
        caution_penalty = 0.12
        corridor_reward_scale = 0.15
    elif caution_mode == "budget":
        caution_penalty = 0.24
        corridor_reward_scale = 0.08
    elif caution_mode == "minimize":
        caution_penalty = 0.45
        corridor_reward_scale = 0.04
    else:  # strict
        caution_penalty = 0.0
        corridor_reward_scale = 0.0
    corridor_reward = max(0.0, float(corridor_bias)) * corridor_reward_scale

    sr, sc = s_rc_adj
    gr, gc = g_rc_adj
    planner_key = planner.strip().lower()
    if planner_key in {"astar", "a_star"}:
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
        )
    elif planner_key in {"dstar_lite", "dstar-lite", "dstar"}:
        cells = _run_dstar_lite_static(
            geo=geo,
            blocked=blocked,
            caution=caution,
            ais_norm=ais_norm,
            start=(sr, sc),
            goal=(gr, gc),
            caution_penalty=caution_penalty,
            corridor_reward=corridor_reward,
        )
    else:
        raise PlanningError(f"Unsupported planner={planner}, expected astar or dstar_lite")

    raw_cells = cells[:]
    if smoothing:
        cells = _smooth_cells_los(cells, blocked)

    coords: list[list[float]] = []
    raw_distance_km = 0.0
    caution_len_km = 0.0
    corridor_vals: list[float] = []
    prev_lat = None
    prev_lon = None
    for idx, (r, c) in enumerate(cells):
        lat, lon = geo.rc_to_latlon(r, c)
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
        "effective_caution_penalty": round(float(caution_penalty), 4),
        "effective_corridor_reward": round(float(corridor_reward), 4),
        "smoothing": bool(smoothing),
        "raw_points": len(raw_cells),
        "smoothed_points": len(cells),
        "start_adjusted": s_rc_adj != s_rc,
        "goal_adjusted": g_rc_adj != g_rc,
        "blocked_ratio": round(float(blocked.mean()), 4),
        "planner": planner_key,
        "grid_bounds": {
            "lat_min": bounds.lat_min,
            "lat_max": bounds.lat_max,
            "lon_min": bounds.lon_min,
            "lon_max": bounds.lon_max,
        },
    }

    route_geojson = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": explain,
    }
    return PlanResult(route_geojson=route_geojson, explain=explain)
