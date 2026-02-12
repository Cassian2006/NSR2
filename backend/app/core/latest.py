from __future__ import annotations

import io
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Callable, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from app.core.config import Settings
from app.core.copernicus_live import CopernicusLiveError, is_copernicus_configured, pull_latest_env_partial
from app.core.geo import load_grid_geo
from app.core.latest_source_health import can_attempt_source, record_source_failure, record_source_success


TIMESTAMP_FMT = "%Y-%m-%d_%H"
DATE_FMT = "%Y-%m-%d"

DEFAULT_CHANNELS = [
    "ice_conc",
    "ice_thick",
    "wave_hs",
    "wind_u10",
    "wind_v10",
    "bathy",
    "ais_heatmap",
]
COPERNICUS_SOURCE = "copernicus_live"
SNAPSHOT_SOURCE = "remote_snapshot"

_TIMESTAMP_LOCKS_LOCK = Lock()
_TIMESTAMP_LOCKS: dict[str, Lock] = {}
_T = TypeVar("_T")


class LatestDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class LatestResolveResult:
    timestamp: str
    source: str
    note: str = ""


ProgressCB = Callable[[str, str, int], None]


def _emit_progress(progress_cb: ProgressCB | None, phase: str, message: str, percent: int) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(phase, message, percent)
    except Exception:
        # Progress reporting should never block core planning flow.
        pass


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_timestamp_lock(timestamp: str) -> Lock:
    with _TIMESTAMP_LOCKS_LOCK:
        lock = _TIMESTAMP_LOCKS.get(timestamp)
        if lock is None:
            lock = Lock()
            _TIMESTAMP_LOCKS[timestamp] = lock
        return lock


def _timestamp_from_date(date_str: str, hour: int) -> str:
    try:
        base = datetime.strptime(date_str, DATE_FMT)
    except ValueError as exc:
        raise LatestDataError(f"Unsupported date format: {date_str}, expected YYYY-MM-DD") from exc
    if hour < 0 or hour > 23:
        raise LatestDataError(f"hour must be in [0, 23], got {hour}")
    return base.replace(hour=hour).strftime(TIMESTAMP_FMT)


def _is_materialized(settings: Settings, timestamp: str) -> bool:
    folder = settings.annotation_pack_root / timestamp
    return (folder / "x_stack.npy").exists() and (folder / "blocked_mask.npy").exists()


def _list_local_timestamps(settings: Settings) -> list[str]:
    items: list[str] = []
    root = settings.annotation_pack_root
    if not root.exists() or not root.is_dir():
        return items
    for folder in sorted(root.iterdir()):
        if folder.is_dir() and (folder / "x_stack.npy").exists() and (folder / "blocked_mask.npy").exists():
            try:
                datetime.strptime(folder.name, TIMESTAMP_FMT)
            except ValueError:
                continue
            items.append(folder.name)
    return items


def _nearest_local_timestamp(settings: Settings, target_timestamp: str) -> str:
    all_ts = _list_local_timestamps(settings)
    if not all_ts:
        raise LatestDataError("No local timestamps available for fallback")
    target_dt = datetime.strptime(target_timestamp, TIMESTAMP_FMT)
    return min(
        all_ts,
        key=lambda ts: abs((datetime.strptime(ts, TIMESTAMP_FMT) - target_dt).total_seconds()),
    )


def _load_channel_names(pack_dir: Path, channels_count: int) -> list[str]:
    meta_path = pack_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            names = meta.get("channel_names")
            if isinstance(names, list) and len(names) == channels_count:
                return [str(v) for v in names]
        except Exception:
            pass
    return DEFAULT_CHANNELS[:channels_count]


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp_path.write_bytes(payload)
    tmp_path.replace(path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    _atomic_write_bytes(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
    )


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    _atomic_write_bytes(path, buffer.getvalue())


def _save_annotation_pack(
    *,
    settings: Settings,
    timestamp: str,
    x_stack: np.ndarray,
    blocked_mask: np.ndarray,
    channel_names: list[str],
    source_meta: dict,
    target_lat: np.ndarray | None = None,
    target_lon: np.ndarray | None = None,
) -> None:
    ann_dir = settings.annotation_pack_root / timestamp
    ann_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_npy(ann_dir / "x_stack.npy", x_stack.astype(np.float32))
    _atomic_write_npy(ann_dir / "blocked_mask.npy", blocked_mask.astype(np.uint8))

    meta = {
        "timestamp": timestamp,
        "x_stack": "x_stack.npy",
        "blocked_mask": "blocked_mask.npy",
        "channel_names": channel_names,
        "shape": [int(x_stack.shape[1]), int(x_stack.shape[2])],
        "source": source_meta,
    }
    if target_lat is not None:
        meta["target_lat"] = [float(v) for v in target_lat.tolist()]
    if target_lon is not None:
        meta["target_lon"] = [float(v) for v in target_lon.tolist()]

    _atomic_write_json(ann_dir / "meta.json", meta)
    _atomic_write_json(ann_dir / "latest_meta.json", source_meta)


def get_latest_meta(settings: Settings, timestamp: str) -> dict:
    p = settings.annotation_pack_root / timestamp / "latest_meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _with_retries(
    *,
    run: Callable[[], _T],
    retries: int,
    backoff_sec: float,
    label: str,
    progress_cb: ProgressCB | None = None,
    phase: str = "download",
    base_percent: int = 30,
) -> _T:
    attempts = max(1, int(retries))
    wait_base = max(0.0, float(backoff_sec))
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return run()
        except Exception as exc:  # pragma: no cover - exercised via call sites
            last_exc = exc
            if attempt >= attempts:
                break
            if not _is_retryable_exception(exc):
                break
            delay = wait_base * (2 ** (attempt - 1))
            delay *= (0.85 + random.random() * 0.30)
            _emit_progress(
                progress_cb,
                phase,
                f"{label} attempt {attempt}/{attempts} failed, retrying in {delay:.1f}s",
                min(95, base_percent + attempt),
            )
            if delay > 0:
                time.sleep(delay)

    if isinstance(last_exc, LatestDataError):
        raise last_exc
    raise LatestDataError(f"{label} failed after {attempts} attempts: {last_exc}")


def _is_retryable_exception(exc: Exception) -> bool:
    # Authentication/config/schema style errors are usually non-retryable.
    text = str(exc).lower()
    non_retry_tokens = [
        "unsupported date format",
        "hour must be",
        "incomplete",
        "missing arrays",
        "shape mismatch",
        "snapshot npz missing required keys",
        "not a valid npz payload",
        "no local timestamps available",
        "auth",
        "invalid credential",
        "unauthorized",
        "forbidden",
        "http 401",
        "http 403",
    ]
    if any(token in text for token in non_retry_tokens):
        return False
    return True


def _download_latest_snapshot(
    *,
    settings: Settings,
    timestamp: str,
    progress_cb: ProgressCB | None = None,
) -> str:
    _emit_progress(progress_cb, "snapshot", "Downloading latest remote snapshot", 35)
    if not settings.latest_snapshot_url_template.strip():
        raise LatestDataError("latest snapshot URL template is not configured")

    date_part, hour_part = timestamp.split("_")
    url = settings.latest_snapshot_url_template.format(
        timestamp=timestamp,
        date=date_part,
        hour=hour_part,
    )
    headers: dict[str, str] = {}
    if settings.latest_snapshot_token.strip():
        headers["Authorization"] = f"Bearer {settings.latest_snapshot_token.strip()}"

    def _download_body() -> bytes:
        req = Request(url=url, headers=headers, method="GET")
        try:
            with urlopen(req, timeout=max(10, int(settings.copernicus_request_timeout_sec))) as resp:
                return resp.read()
        except HTTPError as exc:
            raise LatestDataError(f"latest snapshot fetch failed: HTTP {exc.code}") from exc
        except URLError as exc:
            raise LatestDataError(f"latest snapshot fetch failed: {exc.reason}") from exc

    body = _with_retries(
        run=_download_body,
        retries=settings.latest_remote_retries,
        backoff_sec=settings.latest_remote_retry_backoff_sec,
        label="snapshot download",
        progress_cb=progress_cb,
        phase="snapshot",
        base_percent=38,
    )

    try:
        with np.load(io.BytesIO(body), allow_pickle=False) as npz:
            if "x_stack" not in npz or "blocked_mask" not in npz:
                raise LatestDataError("snapshot npz missing required keys: x_stack/blocked_mask")
            x_stack = np.asarray(npz["x_stack"], dtype=np.float32)
            blocked_mask = np.asarray(npz["blocked_mask"], dtype=np.uint8)
            ais = np.asarray(npz["ais_heatmap"], dtype=np.float32) if "ais_heatmap" in npz else None
            channel_names = (
                [str(v) for v in npz["channel_names"].tolist()]
                if "channel_names" in npz
                else DEFAULT_CHANNELS[: int(x_stack.shape[0])]
            )
    except ValueError as exc:
        raise LatestDataError("snapshot response is not a valid npz payload") from exc

    if x_stack.ndim != 3:
        raise LatestDataError(f"x_stack must be 3D (C,H,W), got {x_stack.shape}")
    if blocked_mask.ndim != 2:
        raise LatestDataError(f"blocked_mask must be 2D (H,W), got {blocked_mask.shape}")
    if x_stack.shape[1:] != blocked_mask.shape:
        raise LatestDataError(f"x_stack HxW {x_stack.shape[1:]} != blocked_mask {blocked_mask.shape}")
    if ais is not None and ais.shape == blocked_mask.shape and x_stack.shape[0] >= 1:
        # If explicit ais_heatmap is provided, place it on the last channel by convention.
        x_stack[-1] = ais

    _save_annotation_pack(
        settings=settings,
        timestamp=timestamp,
        x_stack=x_stack,
        blocked_mask=blocked_mask,
        channel_names=channel_names,
        source_meta={
            "source": "remote_snapshot",
            "snapshot_url": url,
            "materialized_at": _utc_now_iso(),
        },
    )
    _emit_progress(progress_cb, "snapshot", "Remote snapshot persisted", 90)
    return url


def _materialize_from_copernicus(
    *,
    settings: Settings,
    timestamp: str,
    template_timestamp: str | None = None,
    progress_cb: ProgressCB | None = None,
) -> dict:
    if not is_copernicus_configured(settings):
        raise LatestDataError("Copernicus account or dataset settings are incomplete")

    base_timestamp = template_timestamp if template_timestamp and _is_materialized(settings, template_timestamp) else _nearest_local_timestamp(settings, timestamp)
    _emit_progress(progress_cb, "prepare", f"Using template grid {base_timestamp}", 10)
    base_dir = settings.annotation_pack_root / base_timestamp
    x_base_path = base_dir / "x_stack.npy"
    blocked_path = base_dir / "blocked_mask.npy"
    if not x_base_path.exists() or not blocked_path.exists():
        raise LatestDataError(f"Base template timestamp missing arrays: {base_timestamp}")

    base_stack = np.load(x_base_path).astype(np.float32)
    blocked_mask = np.load(blocked_path).astype(np.uint8)
    if base_stack.ndim != 3 or blocked_mask.ndim != 2 or base_stack.shape[1:] != blocked_mask.shape:
        raise LatestDataError(f"Base template shape mismatch at {base_timestamp}")

    channel_names = _load_channel_names(base_dir, channels_count=int(base_stack.shape[0]))
    geo = load_grid_geo(settings=settings, timestamp=base_timestamp, shape=blocked_mask.shape)
    target_time = datetime.strptime(timestamp, TIMESTAMP_FMT)
    _emit_progress(progress_cb, "download", "Pulling Copernicus grids", 20)

    def _pull():
        return pull_latest_env_partial(
            settings=settings,
            target_time=target_time,
            target_lats=geo.lat_axis.astype(np.float64),
            target_lons=geo.lon_axis.astype(np.float64),
            progress_cb=progress_cb,
        )

    try:
        pulled = _with_retries(
            run=_pull,
            retries=settings.latest_remote_retries,
            backoff_sec=settings.latest_remote_retry_backoff_sec,
            label="copernicus pull",
            progress_cb=progress_cb,
            phase="download",
            base_percent=24,
        )
    except CopernicusLiveError as exc:
        raise LatestDataError(str(exc)) from exc
    except LatestDataError as exc:
        raise LatestDataError(str(exc)) from exc

    out_stack = base_stack.copy()
    for idx, ch in enumerate(channel_names):
        field = pulled.fields.get(ch)
        if field is not None and field.shape == blocked_mask.shape:
            out_stack[idx] = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    _emit_progress(progress_cb, "merge", "Merging live channels into model input", 82)

    source_meta = {
        "source": "copernicus_live",
        "materialized_at": _utc_now_iso(),
        "requested_timestamp": timestamp,
        "template_timestamp": base_timestamp,
        "datasets": {
            "ice_dataset_id": settings.copernicus_ice_dataset_id,
            "wave_dataset_id": settings.copernicus_wave_dataset_id,
            "wind_dataset_id": settings.copernicus_wind_dataset_id,
        },
        "variables": {
            "ice_var": settings.copernicus_ice_var,
            "ice_thick_var": settings.copernicus_ice_thick_var,
            "wave_var": settings.copernicus_wave_var,
            "wind_u_var": settings.copernicus_wind_u_var,
            "wind_v_var": settings.copernicus_wind_v_var,
        },
        "channel_source": pulled.channel_source,
        "pulled_channels": sorted(list(pulled.fields.keys())),
        "notes": pulled.notes,
        "stats": pulled.stats,
    }
    if settings.copernicus_wind_u_var.lower() in {"vxo", "uo"} and settings.copernicus_wind_v_var.lower() in {"vyo", "vo"}:
        source_meta["notes"] = list(source_meta.get("notes", [])) + [
            "wind_u10/wind_v10 currently mapped from ocean vector fields (vxo/vyo) as live proxy."
        ]
    _save_annotation_pack(
        settings=settings,
        timestamp=timestamp,
        x_stack=out_stack,
        blocked_mask=blocked_mask,
        channel_names=channel_names,
        source_meta=source_meta,
        target_lat=geo.lat_axis.astype(np.float64),
        target_lon=geo.lon_axis.astype(np.float64),
    )
    _emit_progress(progress_cb, "materialize", "Latest data materialized", 92)
    return source_meta


def resolve_latest_timestamp(
    *,
    settings: Settings,
    date: str,
    hour: int = 12,
    force_refresh: bool = False,
    progress_cb: ProgressCB | None = None,
) -> LatestResolveResult:
    timestamp = _timestamp_from_date(date, hour)
    timestamp_lock = _get_timestamp_lock(timestamp)

    with timestamp_lock:
        has_existing_target = _is_materialized(settings, timestamp)
        if has_existing_target and not force_refresh:
            _emit_progress(progress_cb, "resolve", "Using existing local latest snapshot", 70)
            return LatestResolveResult(timestamp=timestamp, source="local_existing", note="already materialized")

        copernicus_error: str | None = None
        snapshot_error: str | None = None
        if is_copernicus_configured(settings):
            can_use_copernicus, reason = can_attempt_source(COPERNICUS_SOURCE)
            if can_use_copernicus:
                try:
                    source_meta = _materialize_from_copernicus(
                        settings=settings,
                        timestamp=timestamp,
                        template_timestamp=timestamp if has_existing_target else None,
                        progress_cb=progress_cb,
                    )
                    record_source_success(COPERNICUS_SOURCE)
                    _emit_progress(progress_cb, "resolve", "Copernicus pull completed", 94)
                    return LatestResolveResult(
                        timestamp=timestamp,
                        source=COPERNICUS_SOURCE,
                        note=f"materialized from Copernicus ({source_meta.get('materialized_at', '')})",
                    )
                except LatestDataError as exc:
                    # Fall through to snapshot/fallback path.
                    copernicus_error = str(exc)
                    record_source_failure(COPERNICUS_SOURCE, copernicus_error)
            else:
                copernicus_error = f"skipped_by_health_guard:{reason}"

        if settings.latest_snapshot_url_template.strip():
            can_use_snapshot, reason = can_attempt_source(SNAPSHOT_SOURCE)
            if can_use_snapshot:
                try:
                    url = _download_latest_snapshot(settings=settings, timestamp=timestamp, progress_cb=progress_cb)
                    record_source_success(SNAPSHOT_SOURCE)
                    _emit_progress(progress_cb, "resolve", "Remote snapshot pull completed", 94)
                    return LatestResolveResult(timestamp=timestamp, source=SNAPSHOT_SOURCE, note=f"fetched from {url}")
                except LatestDataError as exc:
                    snapshot_error = str(exc)
                    record_source_failure(SNAPSHOT_SOURCE, snapshot_error)
            else:
                snapshot_error = f"skipped_by_health_guard:{reason}"
        else:
            snapshot_error = "snapshot_not_configured"

        if has_existing_target:
            _emit_progress(progress_cb, "resolve", "Refresh failed, falling back to stale local snapshot", 94)
            notes: list[str] = []
            if copernicus_error:
                notes.append(f"copernicus_error={copernicus_error}")
            if snapshot_error:
                notes.append(f"snapshot_error={snapshot_error}")
            note = "; ".join(notes) if notes else "latest sources unavailable"
            return LatestResolveResult(
                timestamp=timestamp,
                source="stale_local_existing",
                note=note,
            )

        fallback = _nearest_local_timestamp(settings, timestamp)
        _emit_progress(progress_cb, "resolve", f"Fallback to nearest local timestamp: {fallback}", 94)

        notes: list[str] = []
        if copernicus_error:
            notes.append(f"copernicus_error={copernicus_error}")
        if snapshot_error:
            notes.append(f"snapshot_error={snapshot_error}")
        note = "; ".join(notes) if notes else "latest sources unavailable"
        return LatestResolveResult(
            timestamp=fallback,
            source="nearest_local_fallback",
            note=note,
        )
