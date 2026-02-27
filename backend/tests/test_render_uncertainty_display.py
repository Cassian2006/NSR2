from __future__ import annotations

import numpy as np

from app.core.render import _enhance_uncertainty_for_display


def test_enhance_uncertainty_boosts_caution_boundary_halo() -> None:
    h, w = 9, 9
    unc = np.full((h, w), 0.12, dtype=np.float32)
    inside = np.ones((h, w), dtype=bool)
    pred = np.zeros((h, w), dtype=np.float32)
    pred[3:6, 3:6] = 1.0  # caution block

    out = _enhance_uncertainty_for_display(unc, inside, pred_sampled=pred)

    # boundary neighborhood around caution should be boosted above background
    assert float(out[2, 4]) > 0.12
    assert float(out[6, 4]) > 0.12
    assert float(out[4, 2]) > 0.12
    assert float(out[4, 6]) > 0.12


def test_enhance_uncertainty_without_pred_keeps_shape_and_bounds() -> None:
    unc = np.array([[0.0, 0.5, 1.2], [-1.0, np.nan, 0.2]], dtype=np.float32)
    inside = np.array([[True, True, True], [True, False, True]], dtype=bool)

    out = _enhance_uncertainty_for_display(unc, inside, pred_sampled=None)
    assert out.shape == unc.shape
    assert float(np.nanmin(out)) >= 0.0
    assert float(np.nanmax(out)) <= 1.0
    assert float(out[1, 1]) == 0.0
