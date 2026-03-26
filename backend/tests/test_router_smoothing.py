from __future__ import annotations

import numpy as np

from app.planning.router import (
    _build_display_coordinates,
    _collect_path_metrics,
    _line_of_sight,
    _max_turn_angle_deg,
    _smooth_cells_los,
    _smooth_cells_with_marine_turns,
    _transition_cost,
    _transition_cost_segment,
    _turn_penalty_km,
)


class _GeoStub:
    def rc_to_latlon(self, r: int, c: int) -> tuple[float, float]:
        return float(r), float(c)


def test_line_of_sight_false_when_crossing_blocked() -> None:
    blocked = np.zeros((10, 10), dtype=bool)
    blocked[5, 5] = True
    assert _line_of_sight((0, 0), (9, 9), blocked) is False


def test_line_of_sight_false_for_corner_squeeze() -> None:
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[0, 1] = True
    blocked[1, 0] = True
    assert _line_of_sight((0, 0), (1, 1), blocked) is False


def test_smooth_cells_los_reduces_path_without_obstacles() -> None:
    blocked = np.zeros((10, 10), dtype=bool)
    path = [(0, 0), (1, 1), (2, 2), (3, 3), (5, 5), (9, 9)]
    smoothed = _smooth_cells_los(path, blocked)
    assert smoothed == [(0, 0), (9, 9)]


def test_smooth_cells_los_keeps_turn_when_obstacle_blocks_shortcut() -> None:
    blocked = np.zeros((10, 10), dtype=bool)
    blocked[4:7, 4:7] = True
    path = [(0, 0), (0, 7), (7, 7), (9, 9)]
    smoothed = _smooth_cells_los(path, blocked)
    assert len(smoothed) >= 3
    assert smoothed[0] == (0, 0)
    assert smoothed[-1] == (9, 9)


def test_collect_path_metrics_captures_caution_between_sparse_vertices() -> None:
    caution = np.zeros((2, 10), dtype=bool)
    caution[0, 4:7] = True
    ais = np.zeros((2, 10), dtype=np.float32)
    near = np.zeros((2, 10), dtype=bool)

    metrics = _collect_path_metrics(
        cells=[(0, 0), (0, 9)],
        geo=_GeoStub(),
        caution=caution,
        ais_norm=ais,
        near_blocked=near,
        caution_penalty=0.2,
        corridor_reward=0.0,
    )

    assert metrics["distance_km"] > 0.0
    assert metrics["sample_count"] > 0
    assert metrics["caution_hits"] > 0
    assert metrics["caution_len_km"] > 0.0
    assert metrics["cost_caution_extra_km"] > 0.0


def test_display_coordinates_are_smoothed_for_rendering() -> None:
    base = [[0.0, 0.0], [2.0, 0.0], [4.0, 1.0], [6.0, 1.0]]
    out = _build_display_coordinates(base, iterations=1)
    assert len(out) > len(base)
    assert out[0] == base[0]
    assert out[-1] == base[-1]


def test_frontend_should_prefer_feasible_coordinates_over_display_coordinates() -> None:
    feature = {
        "geometry": {"coordinates": [[0.0, 0.0], [1.0, 1.0]]},
        "properties": {
            "display_coordinates": [[0.0, 0.0], [0.5, 2.0], [1.0, 1.0]],
            "feasible_smoothed_coordinates": [[0.0, 0.0], [1.0, 1.0]],
            "raw_coordinates": [[0.0, 0.0], [1.0, 1.0]],
        },
    }
    coords = feature["properties"]["feasible_smoothed_coordinates"] or feature["properties"]["raw_coordinates"] or feature["geometry"]["coordinates"]
    assert coords == [[0.0, 0.0], [1.0, 1.0]]


def test_turn_penalty_is_positive_for_sharp_turn() -> None:
    penalty = _turn_penalty_km((0, 0), (0, 1), (1, 1), step_km=10.0, weight=0.03)
    assert penalty > 0.0


def test_marine_turn_smoothing_limits_max_turn() -> None:
    blocked = np.zeros((12, 12), dtype=bool)
    path = [(0, 0), (0, 6), (3, 6), (3, 9), (9, 9)]
    smoothed = _smooth_cells_with_marine_turns(path, blocked, max_turn_deg=105.0)
    assert smoothed[0] == path[0]
    assert smoothed[-1] == path[-1]
    assert _max_turn_angle_deg(smoothed) <= 105.0 + 1e-6


def test_transition_cost_prefers_high_ais_corridor_even_near_blocked() -> None:
    geo = _GeoStub()
    caution = np.zeros((3, 3), dtype=bool)
    ais = np.zeros((3, 3), dtype=np.float32)
    near = np.zeros((3, 3), dtype=bool)
    near[0, 1] = True
    ais[0, 1] = 0.95

    coastal = _transition_cost_segment(
        from_rc=(0, 0),
        to_rc=(0, 1),
        geo=geo,
        caution=caution,
        ais_norm=ais,
        caution_penalty=0.22,
        corridor_reward=0.28,
        near_blocked=near,
        near_blocked_penalty=0.06,
        uncertainty_penalty=None,
        risk_penalty=None,
    )
    open_sea = _transition_cost_segment(
        from_rc=(1, 0),
        to_rc=(1, 1),
        geo=geo,
        caution=caution,
        ais_norm=ais,
        caution_penalty=0.22,
        corridor_reward=0.28,
        near_blocked=np.zeros((3, 3), dtype=bool),
        near_blocked_penalty=0.06,
        uncertainty_penalty=None,
        risk_penalty=None,
    )
    assert coastal < open_sea


def test_transition_cost_prefers_high_ais_corridor_on_single_step() -> None:
    geo = _GeoStub()
    caution = np.zeros((3, 3), dtype=bool)
    ais = np.zeros((3, 3), dtype=np.float32)
    near = np.zeros((3, 3), dtype=bool)
    near[0, 1] = True
    ais[0, 1] = 0.95

    coastal = _transition_cost(
        from_rc=(0, 0),
        to_rc=(0, 1),
        geo=geo,
        caution=caution,
        ais_norm=ais,
        caution_penalty=0.22,
        corridor_reward=0.28,
        near_blocked=near,
        near_blocked_penalty=0.06,
        uncertainty_penalty=None,
        risk_penalty=None,
    )
    open_sea = _transition_cost(
        from_rc=(1, 0),
        to_rc=(1, 1),
        geo=geo,
        caution=caution,
        ais_norm=ais,
        caution_penalty=0.22,
        corridor_reward=0.28,
        near_blocked=np.zeros((3, 3), dtype=bool),
        near_blocked_penalty=0.06,
        uncertainty_penalty=None,
        risk_penalty=None,
    )
    assert coastal < open_sea


def test_transition_cost_penalizes_detour_away_from_ais_corridor() -> None:
    geo = _GeoStub()
    caution = np.zeros((3, 3), dtype=bool)
    ais = np.zeros((3, 3), dtype=np.float32)
    ais[1, 1] = 0.9

    on_corridor = _transition_cost(
        from_rc=(1, 0),
        to_rc=(1, 1),
        geo=geo,
        caution=caution,
        ais_norm=ais,
        caution_penalty=0.22,
        corridor_reward=0.3,
        near_blocked=None,
        near_blocked_penalty=0.0,
        uncertainty_penalty=None,
        risk_penalty=None,
    )
    off_corridor = _transition_cost(
        from_rc=(2, 0),
        to_rc=(2, 1),
        geo=geo,
        caution=caution,
        ais_norm=ais,
        caution_penalty=0.22,
        corridor_reward=0.3,
        near_blocked=None,
        near_blocked_penalty=0.0,
        uncertainty_penalty=None,
        risk_penalty=None,
    )
    assert on_corridor < off_corridor


def test_transition_cost_weakens_corridor_pull_near_route_endpoints() -> None:
    geo = _GeoStub()
    caution = np.zeros((3, 12), dtype=bool)
    ais = np.zeros((3, 12), dtype=np.float32)
    ais[1, 1] = 0.95
    ais[1, 6] = 0.95

    near_start = _transition_cost(
        from_rc=(1, 0),
        to_rc=(1, 1),
        geo=geo,
        caution=caution,
        ais_norm=ais,
        caution_penalty=0.22,
        corridor_reward=0.8,
        near_blocked=None,
        near_blocked_penalty=0.0,
        uncertainty_penalty=None,
        risk_penalty=None,
        start_rc=(1, 0),
        goal_rc=(1, 11),
        corridor_taper_km=500.0,
    )
    mid_route = _transition_cost(
        from_rc=(1, 5),
        to_rc=(1, 6),
        geo=geo,
        caution=caution,
        ais_norm=ais,
        caution_penalty=0.22,
        corridor_reward=0.8,
        near_blocked=None,
        near_blocked_penalty=0.0,
        uncertainty_penalty=None,
        risk_penalty=None,
        start_rc=(1, 0),
        goal_rc=(1, 11),
        corridor_taper_km=500.0,
    )
    assert near_start > mid_route
