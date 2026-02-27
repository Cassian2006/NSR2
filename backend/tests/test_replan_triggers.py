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
        uncertainty_penalty=np.zeros_like(blocked, dtype=np.float32),
        risk_penalty=np.zeros_like(blocked, dtype=np.float32),
    )


def test_dynamic_replan_on_event_reuses_path_when_no_trigger(monkeypatch) -> None:
    h, w = 10, 10
    geo = _GeoStub(h, w)
    blocked = np.zeros((h, w), dtype=bool)
    states = {
        "t0": _state("t0", geo, blocked.copy()),
        "t1": _state("t1", geo, blocked.copy()),
        "t2": _state("t2", geo, blocked.copy()),
    }

    def _fake_load_grid_state(*, settings, timestamp, model_version, blocked_sources, caution_mode, **kwargs):
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
        planner="astar",
        advance_steps=1,
        dynamic_replan_mode="on_event",
        replan_max_skip_steps=99,
        replan_corridor_min=0.0,
        replan_risk_spike=999.0,
    )
    explain = result.explain
    assert explain["dynamic_replan_mode"] == "on_event"
    step_modes = [item.get("update_mode") for item in explain["dynamic_replans"]]
    assert step_modes[0] == "rebuild"
    assert "reuse" in step_modes[1:]
    assert isinstance(explain.get("dynamic_trigger_events"), list)
    assert explain["dynamic_trigger_events"][0]["reasons"] == ["initial_step"]


def test_dynamic_replan_trigger_on_path_blocked(monkeypatch) -> None:
    h, w = 10, 10
    geo = _GeoStub(h, w)
    b0 = np.zeros((h, w), dtype=bool)
    b1 = b0.copy()
    b2 = b1.copy()
    b1[2, 2] = True
    b2[2, 2] = True
    states = {
        "t0": _state("t0", geo, b0),
        "t1": _state("t1", geo, b1),
        "t2": _state("t2", geo, b2),
    }

    def _fake_load_grid_state(*, settings, timestamp, model_version, blocked_sources, caution_mode, **kwargs):
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
        planner="astar",
        advance_steps=1,
        dynamic_replan_mode="on_event",
        replan_blocked_ratio=1.0,  # disable blocked-ratio trigger; force path_blocked trigger
        replan_max_skip_steps=99,
    )
    explain = result.explain
    step1 = next(item for item in explain["dynamic_replans"] if int(item.get("step", -1)) == 1)
    assert bool(step1.get("triggered_replan")) is True
    assert "path_blocked" in list(step1.get("trigger_reasons", []))
