from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.config import Settings


@dataclass(frozen=True)
class GeoBounds:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class GridGeo:
    def __init__(self, lat_axis: np.ndarray, lon_axis: np.ndarray) -> None:
        if lat_axis.ndim != 1 or lon_axis.ndim != 1:
            raise ValueError("lat_axis and lon_axis must be 1D arrays")
        if lat_axis.size < 2 or lon_axis.size < 2:
            raise ValueError("lat_axis and lon_axis must contain at least 2 values")
        self.lat_axis = lat_axis.astype(np.float64)
        self.lon_axis = lon_axis.astype(np.float64)
        self.h = int(self.lat_axis.size)
        self.w = int(self.lon_axis.size)
        self.lat_ascending = bool(self.lat_axis[0] < self.lat_axis[-1])
        self.lon_ascending = bool(self.lon_axis[0] < self.lon_axis[-1])
        self.bounds = GeoBounds(
            lat_min=float(np.min(self.lat_axis)),
            lat_max=float(np.max(self.lat_axis)),
            lon_min=float(np.min(self.lon_axis)),
            lon_max=float(np.max(self.lon_axis)),
        )

    def _idx_from_axis(self, values: np.ndarray, axis: np.ndarray, ascending: bool) -> np.ndarray:
        if ascending:
            idx = np.interp(values, axis, np.arange(axis.size))
        else:
            idx = np.interp(values, axis[::-1], np.arange(axis.size)[::-1])
        return np.rint(idx).astype(np.int64)

    def rows_for_lats(self, lats: np.ndarray) -> np.ndarray:
        rows = self._idx_from_axis(lats.astype(np.float64), self.lat_axis, self.lat_ascending)
        return np.clip(rows, 0, self.h - 1)

    def cols_for_lons(self, lons: np.ndarray) -> np.ndarray:
        cols = self._idx_from_axis(lons.astype(np.float64), self.lon_axis, self.lon_ascending)
        return np.clip(cols, 0, self.w - 1)

    def latlon_to_rc(self, lat: float, lon: float) -> tuple[int, int, bool]:
        inside = self.bounds.lat_min <= lat <= self.bounds.lat_max and self.bounds.lon_min <= lon <= self.bounds.lon_max
        r = int(self.rows_for_lats(np.asarray([lat], dtype=np.float64))[0])
        c = int(self.cols_for_lons(np.asarray([lon], dtype=np.float64))[0])
        return r, c, inside

    def rc_to_latlon(self, r: int, c: int) -> tuple[float, float]:
        rr = min(max(int(r), 0), self.h - 1)
        cc = min(max(int(c), 0), self.w - 1)
        return float(self.lat_axis[rr]), float(self.lon_axis[cc])


def _default_axes(h: int, w: int, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    lat_axis = np.linspace(float(settings.grid_lat_max), float(settings.grid_lat_min), h, dtype=np.float64)
    lon_axis = np.linspace(float(settings.grid_lon_min), float(settings.grid_lon_max), w, dtype=np.float64)
    return lat_axis, lon_axis


def _find_axis_file(pack_dir: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        p = pack_dir / name
        if p.exists():
            return p
    return None


@lru_cache(maxsize=512)
def _load_geo_cached(
    annotation_pack_root: str,
    timestamp: str,
    h: int,
    w: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> GridGeo:
    root = Path(annotation_pack_root)
    pack_dir = root / timestamp
    lat_path = _find_axis_file(pack_dir, ["latitudes.npy", "lats.npy", "lat.npy"])
    lon_path = _find_axis_file(pack_dir, ["longitudes.npy", "lons.npy", "lon.npy"])

    if lat_path is not None and lon_path is not None:
        try:
            lat_axis = np.load(lat_path)
            lon_axis = np.load(lon_path)
            if lat_axis.ndim == 1 and lon_axis.ndim == 1 and lat_axis.size == h and lon_axis.size == w:
                return GridGeo(lat_axis=lat_axis, lon_axis=lon_axis)
        except Exception:
            pass

    lat_axis = np.linspace(lat_max, lat_min, h, dtype=np.float64)
    lon_axis = np.linspace(lon_min, lon_max, w, dtype=np.float64)
    return GridGeo(lat_axis=lat_axis, lon_axis=lon_axis)


def load_grid_geo(settings: Settings, timestamp: str, shape: tuple[int, int]) -> GridGeo:
    h, w = int(shape[0]), int(shape[1])
    return _load_geo_cached(
        annotation_pack_root=str(settings.annotation_pack_root),
        timestamp=timestamp,
        h=h,
        w=w,
        lat_min=float(settings.grid_lat_min),
        lat_max=float(settings.grid_lat_max),
        lon_min=float(settings.grid_lon_min),
        lon_max=float(settings.grid_lon_max),
    )

