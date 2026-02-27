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


def _state(ts: str, geo: _GeoStub, blocked: np.ndarray, caution: np.ndarray, ais: np.ndarray) -> router.GridState:
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
        caution=caution,
        ais_norm=ais,
        free_cells=free_cells,
        uncertainty_penalty=np.zeros_like(ais, dtype=np.float32),
        risk_penalty=np.zeros_like(ais, dtype=np.float32),
    )


def _build_states() -> dict[str, router.GridState]:
    h, w = 12, 12
    geo = _GeoStub(h, w)
    b0 = np.zeros((h, w), dtype=bool)
    b1 = b0.copy()
    b2 = b1.copy()
    b1[4, 4] = True
    b2[4, 5] = True

    c0 = np.zeros((h, w), dtype=bool)
    c1 = c0.copy()
    c2 = c1.copy()
    c1[6, 6] = True
    c2[6, 7] = True

    a0 = np.zeros((h, w), dtype=np.float32)
    a1 = a0.copy()
    a2 = a1.copy()
    a1[3, 3] = 0.5
    a2[3, 4] = 0.5
    return {
        "t0": _state("t0", geo, b0, c0, a0),
        "t1": _state("t1", geo, b1, c1, a1),
        "t2": _state("t2", geo, b2, c2, a2),
    }


def test_dstar_dynamic_reports_changed_edges_and_update_runtime(monkeypatch) -> None:
    states = _build_states()

    def _fake_load_grid_state(*, settings, timestamp, model_version, blocked_sources, caution_mode, **kwargs):
        return states[timestamp]

    monkeypatch.setattr(router, "_load_grid_state", _fake_load_grid_state)
    result = router.plan_grid_route_dynamic(
        settings=get_settings(),
        timestamps=["t0", "t1", "t2"],
        start=(0.0, 0.0),
        goal=(11.0, 11.0),
        model_version="unet_v1",
        corridor_bias=0.0,
        caution_mode="tie_breaker",
        smoothing=False,
        blocked_sources=["bathy"],
        planner="dstar_lite",
        advance_steps=3,
    )
    explain = result.explain
    assert explain["planner"] == "dstar_lite_incremental"
    assert "replan_changed_edges_total" in explain
    assert "replan_update_runtime_ms_total" in explain
    assert explain["replan_full_graph_edges"] > 0
    assert any("changed_edge_count" in item for item in explain["dynamic_replans"])
    assert any(float(item.get("update_runtime_ms", 0.0)) >= 0.0 for item in explain["dynamic_replans"])


def test_dstar_recompute_mode_forces_rebuild(monkeypatch) -> None:
    states = _build_states()

    def _fake_load_grid_state(*, settings, timestamp, model_version, blocked_sources, caution_mode, **kwargs):
        return states[timestamp]

    monkeypatch.setattr(router, "_load_grid_state", _fake_load_grid_state)
    result = router.plan_grid_route_dynamic(
        settings=get_settings(),
        timestamps=["t0", "t1", "t2"],
        start=(0.0, 0.0),
        goal=(11.0, 11.0),
        model_version="unet_v1",
        corridor_bias=0.0,
        caution_mode="tie_breaker",
        smoothing=False,
        blocked_sources=["bathy"],
        planner="dstar_lite_recompute",
        advance_steps=3,
    )
    explain = result.explain
    assert explain["planner"] == "dstar_lite_recompute"
    rebuild_steps = [x for x in explain["dynamic_replans"] if int(x.get("step", 0)) > 0]
    assert rebuild_steps
    assert all(x.get("update_mode") == "rebuild" for x in rebuild_steps)
    assert all(int(x.get("changed_edge_count", 0)) == int(explain["replan_full_graph_edges"]) for x in rebuild_steps)
