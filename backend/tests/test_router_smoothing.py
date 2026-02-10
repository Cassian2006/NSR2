from __future__ import annotations

import numpy as np

from app.planning.router import _line_of_sight, _smooth_cells_los


def test_line_of_sight_false_when_crossing_blocked() -> None:
    blocked = np.zeros((10, 10), dtype=bool)
    blocked[5, 5] = True
    assert _line_of_sight((0, 0), (9, 9), blocked) is False


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

