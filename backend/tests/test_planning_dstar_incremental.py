from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.config import get_settings
from app.planning import router


@dataclass(frozen=True)
class _BoundsStub:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class _GeoStub:
    def __init__(self, h: int, w: int) -> None:
        self.h = h
        self.w = w
        self.bounds = _BoundsStub(lat_min=0.0, lat_max=float(h - 1), lon_min=0.0, lon_max=float(w - 1))

    def rc_to_latlon(self, r: int, c: int) -> tuple[float, float]:
        return float(r), float(c)

    def latlon_to_rc(self, lat: float, lon: float) -> tuple[int, int, bool]:
        r = int(round(lat))
        c = int(round(lon))
        inside = 0 <= r < self.h and 0 <= c < self.w
        return r, c, inside


def _state(ts: str, geo: _GeoStub, blocked: np.ndarray) -> router.GridState:
    free_cells = np.argwhere(~blocked)
    return router.GridState(
        timestamp=ts,
        geo=geo,
        bounds=router.GridBounds(
            lat_min=0.0,
            lat_max=float(blocked.shape[0] - 1),
            lon_min=0.0,
            lon_max=float(blocked.shape[1] - 1),
        ),
        blocked=blocked,
        caution=np.zeros_like(blocked, dtype=bool),
        ais_norm=np.zeros_like(blocked, dtype=np.float32),
        free_cells=free_cells,
    )


def test_dstar_batches_update_and_move_into_single_replan() -> None:
    h, w = 8, 8
    geo = _GeoStub(h, w)
    blocked = np.zeros((h, w), dtype=bool)
    planner = router._DStarLiteIncremental(
        geo=geo,
        blocked=blocked,
        caution=np.zeros((h, w), dtype=bool),
        ais_norm=np.zeros((h, w), dtype=np.float32),
        start=(0, 0),
        goal=(7, 7),
        km_per_row=1.0,
        km_per_col_min=1.0,
        caution_penalty=0.0,
        corridor_reward=0.0,
    )

    calls = {"count": 0}
    original_compute = planner.compute_shortest_path

    def _counted_compute() -> None:
        calls["count"] += 1
        original_compute()

    planner.compute_shortest_path = _counted_compute  # type: ignore[method-assign]

    new_blocked = blocked.copy()
    new_blocked[2, 2] = True
    changed_cells = np.argwhere(blocked != new_blocked)

    planner.update_blocked(new_blocked, changed_cells, auto_compute=False)
    planner.move_start((0, 1), auto_compute=False)
    assert calls["count"] == 0

    planner.sync_if_needed()
    assert calls["count"] == 1
    planner.sync_if_needed()
    assert calls["count"] == 1

    path = planner.extract_path()
    assert path[0] == (0, 1)
    assert path[-1] == (7, 7)


def test_dynamic_route_prefers_incremental_dstar_for_small_changes(monkeypatch) -> None:
    h, w = 10, 10
    geo = _GeoStub(h, w)
    b0 = np.zeros((h, w), dtype=bool)
    b1 = b0.copy()
    b2 = b1.copy()
    b1[4, 4] = True
    b2[4, 5] = True

    states = {
        "t0": _state("t0", geo, b0),
        "t1": _state("t1", geo, b1),
        "t2": _state("t2", geo, b2),
    }

    def _fake_load_grid_state(*, settings, timestamp, model_version, blocked_sources, caution_mode):
        return states[timestamp]

    monkeypatch.setattr(router, "_load_grid_state", _fake_load_grid_state)

    result = router.plan_grid_route_dynamic(
        settings=get_settings(),
        timestamps=["t0", "t1", "t2"],
        start=(0.0, 0.0),
        goal=(9.0, 9.0),
        model_version="unet_v1",
        corridor_bias=0.0,
        caution_mode="tie_breaker",
        smoothing=False,
        blocked_sources=["bathy"],
        planner="dstar_lite",
        advance_steps=2,
    )

    explain = result.explain
    assert explain["planner"] == "dstar_lite_incremental"
    assert explain["dynamic_incremental_steps"] >= 1
    assert any(
        item.get("step", 0) > 0 and item.get("update_mode") == "incremental"
        for item in explain.get("dynamic_replans", [])
    )
