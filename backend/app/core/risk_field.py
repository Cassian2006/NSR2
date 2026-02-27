from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.uncertainty_runtime import calibrate_uncertainty_grid, load_uncertainty_calibration_profile
from app.model.infer import InferenceError, run_unet_inference


RISK_FIELD_VERSION = "risk_v1"
_RISK_LAYERS = {"risk_mean", "risk_p90", "risk_std"}
_WEIGHT_BASE = {
    "unet_pred": 0.34,
    "uncertainty": 0.20,
    "ice": 0.18,
    "wave": 0.14,
    "wind": 0.10,
    "ais_inverse": 0.04,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _risk_dir(settings: Settings) -> Path:
    root = settings.outputs_root / "risk_fields" / RISK_FIELD_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def _risk_paths(settings: Settings, timestamp: str) -> tuple[Path, Path]:
    root = _risk_dir(settings)
    return root / f"{timestamp}.npz", root / f"{timestamp}.meta.json"


def _load_pack(settings: Settings, timestamp: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pack_dir = settings.annotation_pack_root / timestamp
    x_stack = np.load(pack_dir / "x_stack.npy").astype(np.float32)
    blocked = np.load(pack_dir / "blocked_mask.npy").astype(np.float32)
    channels: list[str] = []
    meta_path = pack_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        names = meta.get("channel_names")
        if isinstance(names, list):
            channels = [str(v) for v in names]
    return x_stack, blocked, channels


def _channel_idx(names: list[str], key: str) -> int | None:
    try:
        return names.index(key)
    except ValueError:
        return None


def _load_ais_heatmap(settings: Settings, timestamp: str, shape: tuple[int, int]) -> np.ndarray | None:
    for p in settings.ais_heatmap_root.rglob(f"{timestamp}.npy"):
        try:
            arr = np.load(p).astype(np.float32)
        except Exception:
            continue
        if arr.ndim == 2 and arr.shape == shape:
            return arr
    return None


def _normalize_quantile(values: np.ndarray, sea_mask: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mask = sea_mask & np.isfinite(v)
    if not np.any(mask):
        return np.zeros_like(v, dtype=np.float32)
    sea_values = v[mask]
    lo = float(np.percentile(sea_values, 5))
    hi = float(np.percentile(sea_values, 95))
    if hi - lo < 1e-6:
        return np.zeros_like(v, dtype=np.float32)
    out = (v - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _load_unet_layers(
    *,
    settings: Settings,
    timestamp: str,
    shape: tuple[int, int],
    model_version: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    pred_path = settings.pred_root / model_version / f"{timestamp}.npy"
    unc_path = settings.pred_root / model_version / f"{timestamp}_uncertainty.npy"

    if not pred_path.exists() or not unc_path.exists():
        try:
            run_unet_inference(
                settings=settings,
                timestamp=timestamp,
                model_version=model_version,
                output_path=pred_path,
            )
        except InferenceError:
            pass

    pred = None
    unc = None
    if pred_path.exists():
        try:
            p = np.load(pred_path).astype(np.float32)
            if p.ndim == 2 and p.shape == shape:
                pred = p
        except Exception:
            pred = None
    if unc_path.exists():
        try:
            u = np.load(unc_path).astype(np.float32)
            if u.ndim == 2 and u.shape == shape:
                unc = np.clip(np.nan_to_num(u, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        except Exception:
            unc = None
    return pred, unc


def _compose_components(
    *,
    settings: Settings,
    timestamp: str,
    model_version: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    x_stack, blocked_mask_raw, channels = _load_pack(settings, timestamp)
    if x_stack.ndim != 3:
        raise ValueError(f"x_stack shape must be (C,H,W), got {x_stack.shape}")
    if blocked_mask_raw.ndim != 2:
        raise ValueError(f"blocked_mask shape must be (H,W), got {blocked_mask_raw.shape}")
    if x_stack.shape[1:] != blocked_mask_raw.shape:
        raise ValueError(f"x_stack HxW {x_stack.shape[1:]} != blocked_mask {blocked_mask_raw.shape}")

    h, w = blocked_mask_raw.shape
    blocked_mask = blocked_mask_raw > 0.5
    sea_mask = ~blocked_mask

    components: dict[str, np.ndarray] = {}
    present_sources: list[str] = []
    missing_sources: list[str] = []
    calibration_meta: dict[str, Any] = {}

    pred, unc = _load_unet_layers(
        settings=settings,
        timestamp=timestamp,
        shape=(h, w),
        model_version=model_version,
    )
    if pred is not None:
        pred_component = np.zeros((h, w), dtype=np.float32)
        pred_component[np.rint(pred).astype(np.int16) == 1] = 0.65
        pred_component[np.rint(pred).astype(np.int16) == 2] = 1.0
        components["unet_pred"] = pred_component
        present_sources.append("unet_pred")
    else:
        missing_sources.append("unet_pred")

    if unc is not None:
        profile = load_uncertainty_calibration_profile(settings=settings, model_version=model_version)
        unc_cal = calibrate_uncertainty_grid(unc, temperature=profile.temperature)
        components["uncertainty"] = unc_cal.astype(np.float32)
        present_sources.append("uncertainty")
        calibration_meta = {
            "available": bool(profile.available),
            "temperature": float(profile.temperature),
            "uncertainty_threshold": float(profile.uncertainty_threshold),
            "uplift_scale": float(profile.uplift_scale),
            "source_path": profile.source_path,
            "ece_before": profile.ece_before,
            "ece_after": profile.ece_after,
            "brier_before": profile.brier_before,
            "brier_after": profile.brier_after,
        }
    else:
        missing_sources.append("uncertainty")

    idx_ice = _channel_idx(channels, "ice_conc")
    if idx_ice is not None:
        ice_raw = x_stack[idx_ice]
        ice_norm = ice_raw / 100.0 if float(np.nanmax(np.abs(ice_raw))) > 2.0 else ice_raw
        components["ice"] = np.clip(np.nan_to_num(ice_norm, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
        present_sources.append("ice")
    else:
        missing_sources.append("ice")

    idx_wave = _channel_idx(channels, "wave_hs")
    if idx_wave is not None:
        components["wave"] = _normalize_quantile(x_stack[idx_wave], sea_mask)
        present_sources.append("wave")
    else:
        missing_sources.append("wave")

    idx_wind_u = _channel_idx(channels, "wind_u10")
    idx_wind_v = _channel_idx(channels, "wind_v10")
    if idx_wind_u is not None and idx_wind_v is not None:
        wind_speed = np.sqrt(x_stack[idx_wind_u] ** 2 + x_stack[idx_wind_v] ** 2)
        components["wind"] = _normalize_quantile(wind_speed, sea_mask)
        present_sources.append("wind")
    else:
        missing_sources.append("wind")

    idx_ais = _channel_idx(channels, "ais_heatmap")
    ais = x_stack[idx_ais] if idx_ais is not None else _load_ais_heatmap(settings, timestamp, (h, w))
    if ais is not None and ais.shape == (h, w):
        ais_norm = _normalize_quantile(ais, sea_mask)
        components["ais_inverse"] = np.clip(1.0 - ais_norm, 0.0, 1.0).astype(np.float32)
        present_sources.append("ais_inverse")
    else:
        missing_sources.append("ais_inverse")

    return components, blocked_mask, {
        "present_sources": present_sources,
        "missing_sources": missing_sources,
        "uncertainty_calibration": calibration_meta,
    }


def _fuse_risk(
    components: dict[str, np.ndarray],
    blocked_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    h, w = blocked_mask.shape
    active_keys = [k for k in _WEIGHT_BASE.keys() if k in components]
    if not active_keys:
        risk_mean = np.zeros((h, w), dtype=np.float32)
        risk_std = np.zeros((h, w), dtype=np.float32)
        risk_p90 = np.zeros((h, w), dtype=np.float32)
        risk_mean[blocked_mask] = 1.0
        risk_p90[blocked_mask] = 1.0
        return risk_mean, risk_p90, risk_std, {}

    weight_sum = float(sum(_WEIGHT_BASE[k] for k in active_keys))
    weights = {k: float(_WEIGHT_BASE[k] / max(weight_sum, 1e-8)) for k in active_keys}

    risk_mean = np.zeros((h, w), dtype=np.float32)
    stack_list: list[np.ndarray] = []
    for key in active_keys:
        comp = np.clip(np.nan_to_num(components[key], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
        risk_mean += comp * float(weights[key])
        stack_list.append(comp)

    if len(stack_list) >= 2:
        stack = np.stack(stack_list, axis=0)
        risk_std = np.std(stack, axis=0).astype(np.float32)
    else:
        risk_std = np.zeros((h, w), dtype=np.float32)

    risk_p90 = np.clip(risk_mean + 1.28155 * risk_std, 0.0, 1.0).astype(np.float32)
    risk_mean = np.clip(risk_mean, 0.0, 1.0).astype(np.float32)

    risk_mean[blocked_mask] = 1.0
    risk_p90[blocked_mask] = 1.0
    return risk_mean, risk_p90, risk_std, weights


def compute_risk_fields(
    *,
    settings: Settings,
    timestamp: str,
    model_version: str = "unet_v1",
    force_refresh: bool = False,
) -> dict[str, Any]:
    npz_path, meta_path = _risk_paths(settings, timestamp)
    if not force_refresh and npz_path.exists():
        try:
            with np.load(npz_path) as cached:
                risk_mean = cached["risk_mean"].astype(np.float32)
                risk_p90 = cached["risk_p90"].astype(np.float32)
                risk_std = cached["risk_std"].astype(np.float32)
            meta = {}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["cache_hit"] = True
            return {
                "risk_mean": risk_mean,
                "risk_p90": risk_p90,
                "risk_std": risk_std,
                "meta": meta,
            }
        except Exception:
            pass

    components, blocked_mask, source_meta = _compose_components(
        settings=settings,
        timestamp=timestamp,
        model_version=model_version,
    )
    risk_mean, risk_p90, risk_std, weights = _fuse_risk(components, blocked_mask)

    np.savez_compressed(
        npz_path,
        risk_mean=risk_mean.astype(np.float32),
        risk_p90=risk_p90.astype(np.float32),
        risk_std=risk_std.astype(np.float32),
    )
    meta = {
        "timestamp": timestamp,
        "risk_field_version": RISK_FIELD_VERSION,
        "model_version": model_version,
        "shape": [int(risk_mean.shape[0]), int(risk_mean.shape[1])],
        "weights": weights,
        "present_sources": source_meta["present_sources"],
        "missing_sources": source_meta["missing_sources"],
        "uncertainty_calibration": source_meta.get("uncertainty_calibration", {}),
        "generated_at": _utc_now_iso(),
        "cache_hit": False,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "risk_mean": risk_mean,
        "risk_p90": risk_p90,
        "risk_std": risk_std,
        "meta": meta,
    }


def get_risk_layer(
    *,
    settings: Settings,
    timestamp: str,
    layer: str,
    model_version: str = "unet_v1",
    force_refresh: bool = False,
) -> np.ndarray | None:
    if layer not in _RISK_LAYERS:
        return None
    out = compute_risk_fields(
        settings=settings,
        timestamp=timestamp,
        model_version=model_version,
        force_refresh=force_refresh,
    )
    arr = out.get(layer)
    if isinstance(arr, np.ndarray) and arr.ndim == 2:
        return arr.astype(np.float32)
    return None


def get_risk_summary(
    *,
    settings: Settings,
    timestamp: str,
    model_version: str = "unet_v1",
    force_refresh: bool = False,
) -> dict[str, Any]:
    out = compute_risk_fields(
        settings=settings,
        timestamp=timestamp,
        model_version=model_version,
        force_refresh=force_refresh,
    )
    risk_mean = out["risk_mean"]
    risk_p90 = out["risk_p90"]
    risk_std = out["risk_std"]
    meta = out["meta"]

    def _stats(arr: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "p50": float(np.nanpercentile(arr, 50)),
            "p90": float(np.nanpercentile(arr, 90)),
        }

    return {
        "timestamp": timestamp,
        "risk_field_version": RISK_FIELD_VERSION,
        "shape": [int(risk_mean.shape[0]), int(risk_mean.shape[1])],
        "stats": {
            "risk_mean": _stats(risk_mean),
            "risk_p90": _stats(risk_p90),
            "risk_std": _stats(risk_std),
        },
        "meta": meta,
    }
