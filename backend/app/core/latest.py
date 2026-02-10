from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from app.core.config import Settings
from app.core.copernicus_live import CopernicusLiveError, is_copernicus_configured, pull_latest_env_partial
from app.core.dataset import get_dataset_service
from app.core.geo import load_grid_geo


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


def _nearest_local_timestamp(target_timestamp: str) -> str:
    service = get_dataset_service()
    all_ts = service.list_timestamps(month="all")
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
    np.save(ann_dir / "x_stack.npy", x_stack.astype(np.float32))
    np.save(ann_dir / "blocked_mask.npy", blocked_mask.astype(np.uint8))

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
    (ann_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (ann_dir / "latest_meta.json").write_text(json.dumps(source_meta, ensure_ascii=False, indent=2), encoding="utf-8")


def get_latest_meta(settings: Settings, timestamp: str) -> dict:
    p = settings.annotation_pack_root / timestamp / "latest_meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _download_latest_snapshot(
    *,
    settings: Settings,
    timestamp: str,
    progress_cb: ProgressCB | None = None,
) -> str:
    _emit_progress(progress_cb, "snapshot", "正在下载远程最新快照", 35)
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
    req = Request(url=url, headers=headers, method="GET")
    try:
        with urlopen(req, timeout=60) as resp:
            body = resp.read()
    except HTTPError as exc:
        raise LatestDataError(f"latest snapshot fetch failed: HTTP {exc.code}") from exc
    except URLError as exc:
        raise LatestDataError(f"latest snapshot fetch failed: {exc.reason}") from exc

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
            "materialized_at": datetime.utcnow().isoformat() + "Z",
        },
    )
    _emit_progress(progress_cb, "snapshot", "远程快照已落盘", 90)
    return url


def _materialize_from_copernicus(
    *,
    settings: Settings,
    timestamp: str,
    progress_cb: ProgressCB | None = None,
) -> dict:
    if not is_copernicus_configured(settings):
        raise LatestDataError("Copernicus account or dataset settings are incomplete")

    base_timestamp = _nearest_local_timestamp(timestamp)
    _emit_progress(progress_cb, "prepare", f"使用模板网格 {base_timestamp}", 10)
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
    _emit_progress(progress_cb, "download", "正在拉取 Copernicus 网格", 20)

    try:
        pulled = pull_latest_env_partial(
            settings=settings,
            target_time=target_time,
            target_lats=geo.lat_axis.astype(np.float64),
            target_lons=geo.lon_axis.astype(np.float64),
            progress_cb=progress_cb,
        )
    except CopernicusLiveError as exc:
        raise LatestDataError(str(exc)) from exc

    out_stack = base_stack.copy()
    for idx, ch in enumerate(channel_names):
        field = pulled.fields.get(ch)
        if field is not None and field.shape == blocked_mask.shape:
            out_stack[idx] = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    _emit_progress(progress_cb, "merge", "正在合并实时通道到模型输入", 82)

    source_meta = {
        "source": "copernicus_live",
        "materialized_at": datetime.utcnow().isoformat() + "Z",
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
    _emit_progress(progress_cb, "materialize", "最新数据已物化", 92)
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
    if _is_materialized(settings, timestamp) and not force_refresh:
        _emit_progress(progress_cb, "resolve", "使用已有本地最新快照", 70)
        return LatestResolveResult(timestamp=timestamp, source="local_existing", note="already materialized")

    if is_copernicus_configured(settings):
        try:
            source_meta = _materialize_from_copernicus(
                settings=settings,
                timestamp=timestamp,
                progress_cb=progress_cb,
            )
            _emit_progress(progress_cb, "resolve", "Copernicus 拉取完成", 94)
            return LatestResolveResult(
                timestamp=timestamp,
                source="copernicus_live",
                note=f"materialized from Copernicus ({source_meta.get('materialized_at', '')})",
            )
        except LatestDataError:
            # Fall through to snapshot/fallback path.
            pass

    try:
        url = _download_latest_snapshot(settings=settings, timestamp=timestamp, progress_cb=progress_cb)
        _emit_progress(progress_cb, "resolve", "远程快照拉取完成", 94)
        return LatestResolveResult(timestamp=timestamp, source="remote_snapshot", note=f"fetched from {url}")
    except LatestDataError as exc:
        fallback = _nearest_local_timestamp(timestamp)
        _emit_progress(progress_cb, "resolve", f"回退到最近本地时间片 {fallback}", 94)
        return LatestResolveResult(
            timestamp=fallback,
            source="nearest_local_fallback",
            note=str(exc),
        )
