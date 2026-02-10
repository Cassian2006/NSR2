from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import numpy as np

from app.core.config import Settings


class CopernicusLiveError(RuntimeError):
    pass


@dataclass(frozen=True)
class PulledEnvPartial:
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    stats: dict[str, dict[str, float]] = field(default_factory=dict)
    channel_source: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


ProgressCB = Callable[[str, str, int], None]


def _emit_progress(progress_cb: ProgressCB | None, phase: str, message: str, percent: int) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(phase, message, percent)
    except Exception:
        pass


def is_copernicus_configured(settings: Settings) -> bool:
    """
    Partial configuration is accepted: credentials + at least one dataset id.
    """
    has_auth = bool(settings.copernicus_username.strip() and settings.copernicus_password.strip())
    has_any_ds = bool(
        settings.copernicus_ice_dataset_id.strip()
        or settings.copernicus_wave_dataset_id.strip()
        or settings.copernicus_wind_dataset_id.strip()
    )
    return has_auth and has_any_ds


def _import_optional_deps():
    try:
        import copernicusmarine  # type: ignore[import-not-found]
        import xarray as xr  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise CopernicusLiveError(
            "Copernicus live fetch requires optional dependencies `copernicusmarine` and `xarray`."
        ) from exc
    return copernicusmarine, xr


def _pick_coord_name(candidates: list[str], names: list[str]) -> str | None:
    lowered = {n.lower(): n for n in names}
    for c in candidates:
        hit = lowered.get(c.lower())
        if hit is not None:
            return hit
    return None


def _guess_var_name(data_vars: list[str], fallback: str) -> str:
    if fallback in data_vars:
        return fallback
    lowered = {v.lower(): v for v in data_vars}
    if fallback.lower() in lowered:
        return lowered[fallback.lower()]
    if data_vars:
        return data_vars[0]
    raise CopernicusLiveError("No data variable found in downloaded dataset")


def _normalize_longitudes(da, lon_name: str):
    lon = da[lon_name]
    lon_vals = lon.values.astype(np.float64)
    if np.nanmax(lon_vals) > 180.0:
        lon_new = ((lon_vals + 180.0) % 360.0) - 180.0
        da = da.assign_coords({lon_name: lon_new}).sortby(lon_name)
    return da


def _to_2d_field(
    *,
    xr,
    file_path: Path,
    var_name_hint: str,
    target_time: datetime,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
) -> np.ndarray:
    with xr.open_dataset(file_path) as ds:
        data_vars = list(ds.data_vars)
        var_name = _guess_var_name(data_vars, var_name_hint)
        da = ds[var_name]

        coord_names = list(da.coords) + list(da.dims)
        lat_name = _pick_coord_name(["latitude", "lat", "nav_lat", "y"], coord_names)
        lon_name = _pick_coord_name(["longitude", "lon", "nav_lon", "x"], coord_names)
        if lat_name is None or lon_name is None:
            raise CopernicusLiveError(f"Cannot find lat/lon coords in {file_path.name} for var {var_name}")

        time_name = _pick_coord_name(["time", "valid_time"], coord_names)
        if time_name is not None and time_name in da.dims:
            da = da.sel({time_name: np.datetime64(target_time)}, method="nearest")

        for dim in list(da.dims):
            if dim in {lat_name, lon_name}:
                continue
            if da.sizes.get(dim, 1) > 1:
                da = da.isel({dim: 0})
            else:
                da = da.squeeze(dim=dim, drop=True)

        da = _normalize_longitudes(da, lon_name=lon_name)
        if da[lat_name].values[0] > da[lat_name].values[-1]:
            da = da.sortby(lat_name)

        target_lat_asc = np.asarray(target_lats, dtype=np.float64)
        target_lon_asc = np.asarray(target_lons, dtype=np.float64)
        if target_lat_asc[0] > target_lat_asc[-1]:
            target_lat_asc = target_lat_asc[::-1]

        interp = da.interp(
            {lat_name: target_lat_asc, lon_name: target_lon_asc},
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )
        arr = np.asarray(interp.values, dtype=np.float32)
        if target_lats[0] > target_lats[-1]:
            arr = arr[::-1, :]
        if arr.shape != (target_lats.shape[0], target_lons.shape[0]):
            raise CopernicusLiveError(f"Interpolated field shape mismatch: {arr.shape}")
        return arr


def _download_subset_to_nc(
    *,
    copernicusmarine,
    settings: Settings,
    dataset_id: str,
    variable_names: list[str],
    target_time: datetime,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    out_dir: Path,
    out_file: str,
) -> Path:
    start_dt = target_time
    end_dt = target_time
    kwargs: dict[str, Any] = {
        "dataset_id": dataset_id,
        "variables": variable_names,
        "minimum_longitude": float(lon_min),
        "maximum_longitude": float(lon_max),
        "minimum_latitude": float(lat_min),
        "maximum_latitude": float(lat_max),
        "start_datetime": start_dt.isoformat(),
        "end_datetime": end_dt.isoformat(),
        "output_directory": str(out_dir),
        "output_filename": out_file,
        "username": settings.copernicus_username,
        "password": settings.copernicus_password,
        "force_download": True,
    }
    result = copernicusmarine.subset(**kwargs)

    direct = out_dir / out_file
    if direct.exists():
        return direct
    for attr in ("files", "downloaded_files", "paths"):
        val = getattr(result, attr, None)
        if isinstance(val, list) and val:
            p = Path(val[0])
            if p.exists():
                return p
    hits = sorted(out_dir.glob("*.nc"))
    if hits:
        return hits[0]
    raise CopernicusLiveError(f"No NetCDF output found for dataset {dataset_id}")


def _stats(arr: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(arr)
    if not finite.any():
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    vals = arr[finite]
    return {"min": float(np.min(vals)), "max": float(np.max(vals)), "mean": float(np.mean(vals))}


def pull_latest_env_partial(
    *,
    settings: Settings,
    target_time: datetime,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    progress_cb: ProgressCB | None = None,
) -> PulledEnvPartial:
    if not is_copernicus_configured(settings):
        raise CopernicusLiveError("Copernicus auth or dataset settings are incomplete")

    copernicusmarine, xr = _import_optional_deps()
    lat_min = float(np.min(target_lats))
    lat_max = float(np.max(target_lats))
    lon_min = float(np.min(target_lons))
    lon_max = float(np.max(target_lons))

    fields: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, float]] = {}
    channel_source: dict[str, str] = {}
    notes: list[str] = []

    with TemporaryDirectory(prefix="nsr2_cmems_") as tmp:
        out = Path(tmp)

        requests: list[tuple[str, str, str]] = []
        if settings.copernicus_ice_dataset_id.strip():
            requests.append(("ice_conc", settings.copernicus_ice_dataset_id, settings.copernicus_ice_var))
            requests.append(("ice_thick", settings.copernicus_ice_dataset_id, settings.copernicus_ice_thick_var))
        if settings.copernicus_wave_dataset_id.strip():
            requests.append(("wave_hs", settings.copernicus_wave_dataset_id, settings.copernicus_wave_var))
        if settings.copernicus_wind_dataset_id.strip():
            requests.append(("wind_u10", settings.copernicus_wind_dataset_id, settings.copernicus_wind_u_var))
            requests.append(("wind_v10", settings.copernicus_wind_dataset_id, settings.copernicus_wind_v_var))

        grouped: dict[str, list[tuple[str, str]]] = {}
        for channel_key, dataset_id, var_name in requests:
            if not dataset_id.strip() or not var_name.strip():
                continue
            grouped.setdefault(dataset_id, []).append((channel_key, var_name))

        total_datasets = max(1, len(grouped))
        for dataset_index, (dataset_id, channel_vars) in enumerate(grouped.items(), start=1):
            stage_base = 22 + int((dataset_index - 1) / total_datasets * 46)
            _emit_progress(
                progress_cb,
                "download",
                f"正在下载数据集 {dataset_index}/{total_datasets}: {dataset_id}",
                stage_base,
            )
            unique_vars: list[str] = []
            for _, var_name in channel_vars:
                if var_name not in unique_vars:
                    unique_vars.append(var_name)

            try:
                nc = _download_subset_to_nc(
                    copernicusmarine=copernicusmarine,
                    settings=settings,
                    dataset_id=dataset_id,
                    variable_names=unique_vars,
                    target_time=target_time,
                    lat_min=lat_min,
                    lat_max=lat_max,
                    lon_min=lon_min,
                    lon_max=lon_max,
                    out_dir=out,
                    out_file=f"dataset_{dataset_index}.nc",
                )
            except Exception as exc:
                notes.append(f"dataset pull failed [{dataset_id}]: {exc}")
                _emit_progress(
                    progress_cb,
                    "download",
                    f"数据集下载失败: {dataset_id}",
                    stage_base + 3,
                )
                continue

            for channel_key, var_name in channel_vars:
                try:
                    arr = _to_2d_field(
                        xr=xr,
                        file_path=nc,
                        var_name_hint=var_name,
                        target_time=target_time,
                        target_lats=target_lats,
                        target_lons=target_lons,
                    )
                    fields[channel_key] = arr
                    stats[channel_key] = _stats(arr)
                    channel_source[channel_key] = f"copernicus:{dataset_id}/{var_name}"
                    _emit_progress(
                        progress_cb,
                        "download",
                        f"已映射通道 {channel_key}",
                        stage_base + 4,
                    )
                except Exception as exc:
                    notes.append(f"{channel_key} pull failed [{dataset_id}/{var_name}]: {exc}")
            stage_done = 22 + int(dataset_index / total_datasets * 46)
            _emit_progress(
                progress_cb,
                "download",
                f"数据集处理完成 {dataset_index}/{total_datasets}",
                stage_done,
            )

    if not fields:
        raise CopernicusLiveError("No live channels pulled from configured Copernicus datasets")

    return PulledEnvPartial(fields=fields, stats=stats, channel_source=channel_source, notes=notes)
