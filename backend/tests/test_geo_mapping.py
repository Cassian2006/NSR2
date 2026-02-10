from __future__ import annotations

import numpy as np

from app.core.config import Settings
from app.core.geo import GridGeo, load_grid_geo


def test_grid_geo_axis_mapping_roundtrip() -> None:
    lat_axis = np.asarray([86.0, 83.0, 80.0, 77.0], dtype=np.float64)
    lon_axis = np.asarray([-10.0, 0.0, 12.0, 20.0], dtype=np.float64)
    geo = GridGeo(lat_axis=lat_axis, lon_axis=lon_axis)

    r, c, inside = geo.latlon_to_rc(82.9, 0.5)
    assert inside
    assert r == 1
    assert c == 1

    lat, lon = geo.rc_to_latlon(r, c)
    assert lat == 83.0
    assert lon == 0.0


def test_load_grid_geo_prefers_annotation_axes(tmp_path) -> None:
    ts = "2024-07-01_00"
    pack = tmp_path / ts
    pack.mkdir(parents=True, exist_ok=True)
    np.save(pack / "latitudes.npy", np.asarray([90.0, 80.0, 70.0], dtype=np.float64))
    np.save(pack / "longitudes.npy", np.asarray([-50.0, -10.0, 20.0, 40.0], dtype=np.float64))

    settings = Settings(
        annotation_pack_root=tmp_path,
        env_grids_root=tmp_path / "env_grids",
        grid_lat_min=60.0,
        grid_lat_max=86.0,
        grid_lon_min=-180.0,
        grid_lon_max=180.0,
    )
    geo = load_grid_geo(settings, timestamp=ts, shape=(3, 4))
    assert geo.bounds.lat_max == 90.0
    assert geo.bounds.lat_min == 70.0
    assert geo.bounds.lon_min == -50.0
    assert geo.bounds.lon_max == 40.0


def test_load_grid_geo_fallback_to_configured_bounds(tmp_path) -> None:
    ts = "2024-07-02_00"
    (tmp_path / ts).mkdir(parents=True, exist_ok=True)
    settings = Settings(
        annotation_pack_root=tmp_path,
        env_grids_root=tmp_path / "env_grids",
        grid_lat_min=61.0,
        grid_lat_max=85.0,
        grid_lon_min=-160.0,
        grid_lon_max=170.0,
    )
    geo = load_grid_geo(settings, timestamp=ts, shape=(5, 6))
    assert geo.bounds.lat_min == 61.0
    assert geo.bounds.lat_max == 85.0
    assert geo.bounds.lon_min == -160.0
    assert geo.bounds.lon_max == 170.0
