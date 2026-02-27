from __future__ import annotations

import numpy as np

from app.core.render import _render_bathy


def test_render_bathy_uses_soft_alpha_transition() -> None:
    sampled = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    inside = np.array([[True, True, True]], dtype=bool)

    out = _render_bathy(sampled, inside)
    a0, a1, a2 = int(out[0, 0, 3]), int(out[0, 1, 3]), int(out[0, 2, 3])

    assert a0 == 0
    assert 0 < a1 < a2 <= 190
    assert tuple(out[0, 2, :3]) == (10, 10, 10)


def test_render_bathy_respects_inside_mask() -> None:
    sampled = np.array([[1.0, 1.0]], dtype=np.float32)
    inside = np.array([[True, False]], dtype=bool)

    out = _render_bathy(sampled, inside)
    assert int(out[0, 0, 3]) > 0
    assert tuple(out[0, 1]) == (0, 0, 0, 0)
