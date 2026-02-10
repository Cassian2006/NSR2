from __future__ import annotations

import numpy as np

from app.core.render import _render_unet


def test_render_unet_keeps_caution_and_blocked_colors() -> None:
    sampled = np.array([[0, 1, 2]], dtype=np.int16)
    inside = np.array([[True, True, True]], dtype=bool)

    out = _render_unet(sampled, inside)
    assert tuple(out[0, 0]) == (0, 0, 0, 0)
    assert tuple(out[0, 1]) == (245, 158, 11, 145)
    assert tuple(out[0, 2]) == (239, 68, 68, 185)


def test_render_unet_hides_predictions_on_bathy_blocked_cells() -> None:
    sampled = np.array([[1, 2, 2]], dtype=np.int16)
    inside = np.array([[True, True, True]], dtype=bool)
    bathy = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)

    out = _render_unet(sampled, inside, bathy_blocked_sampled=bathy)
    assert tuple(out[0, 0]) == (0, 0, 0, 0)
    assert tuple(out[0, 1]) == (0, 0, 0, 0)
    assert tuple(out[0, 2]) == (239, 68, 68, 185)
