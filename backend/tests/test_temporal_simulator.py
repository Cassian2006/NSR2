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


def _build_states() -> dict[str, router.GridState]:
    h, w = 12, 12
    geo = _GeoStub(h, w)
    b0 = np.zeros((h, w), dtype=bool)
    b1 = b0.copy()
    b2 = b1.copy()
    b1[5, 5] = True
    b2[6, 6] = True
    return {
        "t0": _state("t0", geo, b0),
        "t1": _state("t1", geo, b1),
        "t2": _state("t2", geo, b2),
    }


def _run_dynamic(monkeypatch):
    states = _build_states()

    def _fake_load_grid_state(*, settings, timestamp, model_version, blocked_sources, caution_mode, **kwargs):
        return states[timestamp]

    monkeypatch.setattr(router, "_load_grid_state", _fake_load_grid_state)
    return router.plan_grid_route_dynamic(
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
        advance_steps=2,
        dynamic_replan_mode="on_event",
    )


def test_temporal_simulator_outputs_execution_log(monkeypatch) -> None:
    result = _run_dynamic(monkeypatch)
    explain = result.explain
    log = explain.get("dynamic_execution_log", [])
    assert isinstance(log, list)
    assert len(log) == len(explain.get("dynamic_replans", []))
    assert explain.get("dynamic_simulation_mode") == "advance_replan_loop"
    assert explain.get("dynamic_replay_ready") is True
    assert explain.get("dynamic_execution_steps") == len(log)
    assert float(explain.get("dynamic_cumulative_distance_km", 0.0)) == float(explain.get("distance_km", 0.0))
    assert float(explain.get("dynamic_cumulative_risk_extra_km", 0.0)) == float(explain.get("route_cost_risk_extra_km", 0.0))

    prev_dist = -1.0
    for item in log:
        assert "timestamp" in item
        assert "segment_coordinates" in item
        assert isinstance(item["segment_coordinates"], list)
        assert len(item["segment_coordinates"]) >= 2
        cur = float(item.get("cumulative_distance_km", 0.0))
        assert cur >= prev_dist
        prev_dist = cur


def test_temporal_simulator_reproducible_without_randomness(monkeypatch) -> None:
    result1 = _run_dynamic(monkeypatch)
    result2 = _run_dynamic(monkeypatch)
    assert result1.route_geojson["geometry"]["coordinates"] == result2.route_geojson["geometry"]["coordinates"]

    def _normalize(log: list[dict]) -> list[dict]:
        out: list[dict] = []
        for item in log:
            out.append(
                {
                    "step": int(item.get("step", -1)),
                    "timestamp": str(item.get("timestamp", "")),
                    "update_mode": str(item.get("update_mode", "")),
                    "trigger_reasons": list(item.get("trigger_reasons", [])),
                    "moved_edges": int(item.get("moved_edges", 0)),
                    "moved_distance_km": float(item.get("moved_distance_km", 0.0)),
                    "cumulative_distance_km": float(item.get("cumulative_distance_km", 0.0)),
                    "cumulative_risk_extra_km": float(item.get("cumulative_risk_extra_km", 0.0)),
                    "segment_start": list(item.get("segment_start", [])),
                    "segment_end": list(item.get("segment_end", [])),
                }
            )
        return out

    log1 = _normalize(result1.explain.get("dynamic_execution_log", []))
    log2 = _normalize(result2.explain.get("dynamic_execution_log", []))
    assert log1 == log2
