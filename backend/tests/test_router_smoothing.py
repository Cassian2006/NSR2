from __future__ import annotations

import numpy as np

from app.planning.router import _build_display_coordinates, _collect_path_metrics, _line_of_sight, _smooth_cells_los


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
