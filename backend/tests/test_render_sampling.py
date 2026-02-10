from __future__ import annotations

import numpy as np

from app.core.geo import GridGeo
from app.core.render import _normalize_ice_sampled, _render_continuous, _sample_grid_from_axes


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


def test_render_continuous_respects_value_mask() -> None:
    sampled = np.asarray([[0.2, 0.8]], dtype=np.float32)
    inside = np.asarray([[True, True]], dtype=bool)
    value_mask = np.asarray([[False, True]], dtype=bool)

    image = _render_continuous(
        sampled,
        inside,
        stops=[(0.0, (0, 0, 255)), (1.0, (255, 0, 0))],
        alpha_min=20,
        alpha_max=120,
        value_mask=value_mask,
    )
    assert int(image[0, 0, 3]) == 0
    assert int(image[0, 1, 3]) > 0


def test_normalize_ice_sampled_fills_non_finite_as_zero() -> None:
    sampled = np.asarray([[np.nan, 12.0, -4.0, np.inf]], dtype=np.float32)
    out = _normalize_ice_sampled(sampled)
    assert np.all(np.isfinite(out))
    assert np.isclose(float(out[0, 0]), 0.0)
    assert np.isclose(float(out[0, 1]), 12.0)
    assert np.isclose(float(out[0, 2]), 0.0)
    assert np.isclose(float(out[0, 3]), 0.0)
