from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from app.core.config import Settings


class StormglassLiveError(RuntimeError):
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


def is_stormglass_configured(settings: Settings) -> bool:
    return bool(settings.stormglass_api_key.strip())


def _stats(arr: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(arr)
    if not finite.any():
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    vals = arr[finite]
    return {"min": float(np.min(vals)), "max": float(np.max(vals)), "mean": float(np.mean(vals))}


def _pick_source_value(payload: dict, preferred: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    if preferred in payload:
        val = payload.get(preferred)
        if isinstance(val, (int, float)):
            return float(val)
    if "sg" in payload:
        val = payload.get("sg")
        if isinstance(val, (int, float)):
            return float(val)
    for val in payload.values():
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _wind_to_uv(speed: float, direction_deg: float) -> tuple[float, float]:
    # Stormglass direction follows meteorological convention (coming from, clockwise from north).
    radians = math.radians(float(direction_deg))
    u = -float(speed) * math.sin(radians)
    v = -float(speed) * math.cos(radians)
    return float(u), float(v)


def _interp_regular_grid(
    *,
    values: np.ndarray,
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
) -> np.ndarray:
    lon_asc = bool(src_lons[0] <= src_lons[-1])
    lat_asc = bool(src_lats[0] <= src_lats[-1])
    if not lon_asc:
        src_lons = src_lons[::-1].copy()
        values = values[:, ::-1].copy()
    if not lat_asc:
        src_lats = src_lats[::-1].copy()
        values = values[::-1, :].copy()

    target_lon_asc = target_lons if target_lons[0] <= target_lons[-1] else target_lons[::-1]
    target_lat_asc = target_lats if target_lats[0] <= target_lats[-1] else target_lats[::-1]

    tmp = np.empty((values.shape[0], target_lon_asc.size), dtype=np.float32)
    for i in range(values.shape[0]):
        tmp[i, :] = np.interp(target_lon_asc, src_lons, values[i, :]).astype(np.float32)

    out_asc = np.empty((target_lat_asc.size, target_lon_asc.size), dtype=np.float32)
    for j in range(tmp.shape[1]):
        out_asc[:, j] = np.interp(target_lat_asc, src_lats, tmp[:, j]).astype(np.float32)

    out = out_asc
    if target_lats[0] > target_lats[-1]:
        out = out[::-1, :]
    if target_lons[0] > target_lons[-1]:
        out = out[:, ::-1]
    return out.astype(np.float32)


def _fill_nan_grid(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float32).copy()
    finite = np.isfinite(out)
    if finite.all():
        return out
    if not finite.any():
        return np.zeros_like(out, dtype=np.float32)
    mean_val = float(np.nanmean(out))
    out[~finite] = mean_val
    return out


def _cache_file(settings: Settings, target_time: datetime) -> Path:
    stamp = target_time.astimezone(timezone.utc).strftime("%Y-%m-%d_%H")
    return settings.stormglass_cache_root / f"{stamp}_{settings.stormglass_sample_lat_count}x{settings.stormglass_sample_lon_count}.npz"


def _build_url(lat: float, lon: float, target_time: datetime) -> str:
    start = target_time.astimezone(timezone.utc)
    end = start + timedelta(hours=1)
    params = {
        "lat": f"{lat:.6f}",
        "lng": f"{lon:.6f}",
        "params": "waveHeight,windSpeed,windDirection",
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    return "https://api.stormglass.io/v2/weather/point?" + urlencode(params)


def _fetch_point(settings: Settings, *, lat: float, lon: float, target_time: datetime) -> dict:
    url = _build_url(lat=lat, lon=lon, target_time=target_time)
    req = Request(
        url=url,
        headers={"Authorization": settings.stormglass_api_key.strip()},
        method="GET",
    )
    try:
        with urlopen(req, timeout=max(5, int(settings.stormglass_request_timeout_sec))) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = ""
        try:
            body = exc.read().decode("utf-8")
            payload = json.loads(body)
            if isinstance(payload, dict):
                errors = payload.get("errors", {})
                meta = payload.get("meta", {})
                if isinstance(errors, dict):
                    detail = str(errors.get("key", "")) or str(errors)
                if isinstance(meta, dict) and meta.get("dailyQuota") is not None:
                    detail = f"{detail}; dailyQuota={meta.get('dailyQuota')}, requestCount={meta.get('requestCount')}"
        except Exception:
            pass
        raise StormglassLiveError(f"Stormglass fetch failed: HTTP {exc.code} {detail}".strip()) from exc
    except URLError as exc:
        raise StormglassLiveError(f"Stormglass fetch failed: {exc.reason}") from exc
    except Exception as exc:
        raise StormglassLiveError(f"Stormglass fetch failed at lat={lat:.3f}, lon={lon:.3f}: {exc}") from exc

    if isinstance(payload, dict) and payload.get("errors"):
        raise StormglassLiveError(f"Stormglass returned errors: {payload.get('errors')}")

    hours = payload.get("hours")
    if not isinstance(hours, list) or not hours:
        raise StormglassLiveError(f"Stormglass response missing hours at lat={lat:.3f}, lon={lon:.3f}")

    return hours[0]


def pull_latest_env_partial(
    *,
    settings: Settings,
    target_time: datetime,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    progress_cb: ProgressCB | None = None,
) -> PulledEnvPartial:
    if not is_stormglass_configured(settings):
        raise StormglassLiveError("Stormglass API key is not configured")

    cache_path = _cache_file(settings, target_time.replace(tzinfo=timezone.utc))
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as npz:
            fields = {
                "wave_hs": np.asarray(npz["wave_hs"], dtype=np.float32),
                "wind_u10": np.asarray(npz["wind_u10"], dtype=np.float32),
                "wind_v10": np.asarray(npz["wind_v10"], dtype=np.float32),
            }
            stats = json.loads(str(npz["stats_json"].tolist()))
            channel_source = json.loads(str(npz["channel_source_json"].tolist()))
            notes = json.loads(str(npz["notes_json"].tolist()))
        _emit_progress(progress_cb, "download", "Using cached Stormglass snapshot", 36)
        return PulledEnvPartial(fields=fields, stats=stats, channel_source=channel_source, notes=notes)

    lat_min = float(np.min(target_lats))
    lat_max = float(np.max(target_lats))
    lon_min = float(np.min(target_lons))
    lon_max = float(np.max(target_lons))
    sample_lats = np.linspace(lat_min, lat_max, max(2, int(settings.stormglass_sample_lat_count)), dtype=np.float64)
    sample_lons = np.linspace(lon_min, lon_max, max(2, int(settings.stormglass_sample_lon_count)), dtype=np.float64)

    wave_src = np.zeros((sample_lats.size, sample_lons.size), dtype=np.float32)
    wind_u_src = np.zeros_like(wave_src)
    wind_v_src = np.zeros_like(wave_src)

    total = int(sample_lats.size * sample_lons.size)
    completed = 0
    futures = {}
    max_workers = min(6, total)
    preferred = settings.stormglass_source_preference.strip() or "sg"
    _emit_progress(progress_cb, "download", "Pulling Stormglass point samples", 22)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, lat in enumerate(sample_lats):
            for j, lon in enumerate(sample_lons):
                futures[pool.submit(_fetch_point, settings, lat=float(lat), lon=float(lon), target_time=target_time.replace(tzinfo=timezone.utc))] = (i, j)

        for fut in as_completed(futures):
            i, j = futures[fut]
            hour = fut.result()
            wave = _pick_source_value(hour.get("waveHeight", {}), preferred)
            wind_speed = _pick_source_value(hour.get("windSpeed", {}), preferred)
            wind_dir = _pick_source_value(hour.get("windDirection", {}), preferred)
            wave_src[i, j] = np.nan if wave is None else float(wave)
            if wind_speed is None or wind_dir is None:
                wind_u_src[i, j] = np.nan
                wind_v_src[i, j] = np.nan
            else:
                wind_u_src[i, j], wind_v_src[i, j] = _wind_to_uv(float(wind_speed), float(wind_dir))
            completed += 1
            percent = 22 + int(48 * completed / max(1, total))
            _emit_progress(progress_cb, "download", f"Stormglass samples {completed}/{total}", percent)

    wave_src = _fill_nan_grid(wave_src)
    wind_u_src = _fill_nan_grid(wind_u_src)
    wind_v_src = _fill_nan_grid(wind_v_src)

    wave_hs = _interp_regular_grid(
        values=wave_src,
        src_lats=sample_lats,
        src_lons=sample_lons,
        target_lats=target_lats.astype(np.float64),
        target_lons=target_lons.astype(np.float64),
    )
    wind_u10 = _interp_regular_grid(
        values=wind_u_src,
        src_lats=sample_lats,
        src_lons=sample_lons,
        target_lats=target_lats.astype(np.float64),
        target_lons=target_lons.astype(np.float64),
    )
    wind_v10 = _interp_regular_grid(
        values=wind_v_src,
        src_lats=sample_lats,
        src_lons=sample_lons,
        target_lats=target_lats.astype(np.float64),
        target_lons=target_lons.astype(np.float64),
    )

    fields = {
        "wave_hs": wave_hs,
        "wind_u10": wind_u10,
        "wind_v10": wind_v10,
    }
    stats = {key: _stats(arr) for key, arr in fields.items()}
    channel_source = {
        "wave_hs": "stormglass.waveHeight",
        "wind_u10": "stormglass.windSpeed+windDirection",
        "wind_v10": "stormglass.windSpeed+windDirection",
    }
    notes = [
        f"stormglass_sample_grid={sample_lats.size}x{sample_lons.size}",
        "ice channels remain from local template; Stormglass currently refreshes wave and wind only.",
    ]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        wave_hs=wave_hs.astype(np.float32),
        wind_u10=wind_u10.astype(np.float32),
        wind_v10=wind_v10.astype(np.float32),
        stats_json=json.dumps(stats, ensure_ascii=False),
        channel_source_json=json.dumps(channel_source, ensure_ascii=False),
        notes_json=json.dumps(notes, ensure_ascii=False),
    )
    _emit_progress(progress_cb, "merge", "Stormglass fields interpolated onto model grid", 78)
    return PulledEnvPartial(fields=fields, stats=stats, channel_source=channel_source, notes=notes)
