from __future__ import annotations

import numpy as np

from app.core.geo import GridGeo
from app.core.render import _sample_grid_from_axes


def test_sample_grid_bilinear_interpolates_continuous_values() -> None:
    data = np.asarray([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    geo = GridGeo(
        lat_axis=np.asarray([1.0, 0.0], dtype=np.float64),
        lon_axis=np.asarray([0.0, 1.0], dtype=np.float64),
    )

    sampled, inside = _sample_grid_from_axes(
        data,
        lats=np.asarray([0.5], dtype=np.float64),
        lons=np.asarray([0.5], dtype=np.float64),
        geo=geo,
        mode="bilinear",
    )
    assert inside.shape == (1, 1)
    assert inside[0, 0]
    assert np.isclose(float(sampled[0, 0]), 15.0, atol=1e-5)


def test_sample_grid_nearest_for_discrete_layers() -> None:
    data = np.asarray([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    geo = GridGeo(
        lat_axis=np.asarray([1.0, 0.0], dtype=np.float64),
        lon_axis=np.asarray([0.0, 1.0], dtype=np.float64),
    )

    sampled, inside = _sample_grid_from_axes(
        data,
        lats=np.asarray([0.9], dtype=np.float64),
        lons=np.asarray([0.9], dtype=np.float64),
        geo=geo,
        mode="nearest",
    )
    assert inside[0, 0]
    assert float(sampled[0, 0]) == 10.0
