from __future__ import annotations

import numpy as np

from app.planning.router import _collect_path_metrics, _transition_cost


class _GeoStub:
    def rc_to_latlon(self, r: int, c: int) -> tuple[float, float]:
        return float(r), float(c)


def test_transition_cost_increases_in_high_uncertainty_cell() -> None:
    geo = _GeoStub()
    caution = np.zeros((3, 3), dtype=bool)
    ais = np.zeros((3, 3), dtype=np.float32)
    near = np.zeros((3, 3), dtype=bool)
    unc_penalty = np.zeros((3, 3), dtype=np.float32)
    unc_penalty[1, 1] = 0.45

    base = _transition_cost(
        from_rc=(1, 0),
        to_rc=(1, 1),
        geo=geo,
        caution=caution,
        ais_norm=ais,
        caution_penalty=0.2,
        corridor_reward=0.0,
        near_blocked=near,
        near_blocked_penalty=0.0,
        uncertainty_penalty=None,
    )
    boosted = _transition_cost(
        from_rc=(1, 0),
        to_rc=(1, 1),
        geo=geo,
        caution=caution,
        ais_norm=ais,
        caution_penalty=0.2,
        corridor_reward=0.0,
        near_blocked=near,
        near_blocked_penalty=0.0,
        uncertainty_penalty=unc_penalty,
    )
    assert boosted > base


def test_collect_path_metrics_tracks_uncertainty_extra_cost() -> None:
    geo = _GeoStub()
    cells = [(1, 0), (1, 1), (1, 2)]
    caution = np.zeros((3, 3), dtype=bool)
    ais = np.zeros((3, 3), dtype=np.float32)
    near = np.zeros((3, 3), dtype=bool)
    unc_penalty = np.zeros((3, 3), dtype=np.float32)
    unc_penalty[1, 1] = 0.3
    unc_penalty[1, 2] = 0.4

    m = _collect_path_metrics(
        cells=cells,
        geo=geo,
        caution=caution,
        ais_norm=ais,
        near_blocked=near,
        caution_penalty=0.0,
        corridor_reward=0.0,
        uncertainty_penalty=unc_penalty,
    )
    assert float(m["cost_uncertainty_extra_km"]) > 0.0
    vals = list(m["uncertainty_penalty_vals"])
    assert len(vals) >= 2
    assert max(vals) >= 0.3
