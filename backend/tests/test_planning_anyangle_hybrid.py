from __future__ import annotations

import numpy as np

from app.planning.router import _line_of_sight, _run_astar, _run_hybrid_astar, _run_theta_star


class _GeoStub:
    def rc_to_latlon(self, r: int, c: int) -> tuple[float, float]:
        return float(r), float(c)


def test_theta_star_is_not_more_vertex_dense_than_astar_on_open_grid() -> None:
    h, w = 30, 30
    blocked = np.zeros((h, w), dtype=bool)
    caution = np.zeros((h, w), dtype=bool)
    ais = np.zeros((h, w), dtype=np.float32)
    near = np.zeros((h, w), dtype=bool)
    geo = _GeoStub()
    start = (2, 2)
    goal = (27, 26)

    astar = _run_astar(
        geo=geo,
        blocked=blocked,
        caution=caution,
        ais_norm=ais,
        start=start,
        goal=goal,
        km_per_row=1.0,
        km_per_col_min=1.0,
        caution_penalty=0.0,
        corridor_reward=0.0,
        near_blocked=near,
        near_blocked_penalty=0.0,
    )
    theta = _run_theta_star(
        geo=geo,
        blocked=blocked,
        caution=caution,
        ais_norm=ais,
        start=start,
        goal=goal,
        km_per_row=1.0,
        km_per_col_min=1.0,
        caution_penalty=0.0,
        corridor_reward=0.0,
        near_blocked=near,
        near_blocked_penalty=0.0,
    )
    assert theta[0] == start
    assert theta[-1] == goal
    assert len(theta) <= len(astar)


def test_hybrid_astar_returns_feasible_path_with_obstacle_gap() -> None:
    h, w = 40, 40
    blocked = np.zeros((h, w), dtype=bool)
    blocked[10:34, 19] = True
    blocked[21, 19] = False  # gap
    caution = np.zeros((h, w), dtype=bool)
    ais = np.zeros((h, w), dtype=np.float32)
    near = np.zeros((h, w), dtype=bool)
    geo = _GeoStub()
    start = (8, 8)
    goal = (34, 30)

    path = _run_hybrid_astar(
        geo=geo,
        blocked=blocked,
        caution=caution,
        ais_norm=ais,
        start=start,
        goal=goal,
        km_per_row=1.0,
        km_per_col_min=1.0,
        caution_penalty=0.0,
        corridor_reward=0.0,
        near_blocked=near,
        near_blocked_penalty=0.0,
    )
    assert path[0] == start
    assert path[-1] == goal
    assert len(path) >= 2
    for idx in range(1, len(path)):
        pr, pc = path[idx - 1]
        cr, cc = path[idx]
        assert not blocked[cr, cc]
        assert _line_of_sight((pr, pc), (cr, cc), blocked)

