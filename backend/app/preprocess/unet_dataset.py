from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TIMESTAMP_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}$")


@dataclass(frozen=True)
class FeatureBundle:
    stack: np.ndarray
    channel_names: list[str]
    has_bathy: bool


def list_sample_timestamps(samples_root: Path, months: set[str] | None = None) -> list[str]:
    found: set[str] = set()
    if not samples_root.exists():
        return []

    for month_dir in sorted(samples_root.iterdir()):
        if not month_dir.is_dir():
            continue
        month_name = month_dir.name
        if not (len(month_name) == 6 and month_name.isdigit()):
            continue
        if months is not None and month_name not in months:
            continue
        for sample_dir in sorted(month_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            ts = sample_dir.name
            if not TIMESTAMP_DIR_RE.match(ts):
                continue
            if not (sample_dir / "y_corridor.npy").exists():
                continue
            found.add(ts)
    return sorted(found)


def _read_env_channel_names(env_dir: Path, expected_channels: int) -> list[str]:
    meta_path = env_dir / "meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        channels = meta.get("channels")
        if isinstance(channels, list) and len(channels) == expected_channels:
            return [str(v) for v in channels]
    return [f"env_{idx}" for idx in range(expected_channels)]


def load_feature_stack(
    timestamp: str,
    env_root: Path,
    heatmap_root: Path,
    bathy_fill_value: float = 0.0,
) -> FeatureBundle:
    if not TIMESTAMP_DIR_RE.match(timestamp):
        raise ValueError(f"Invalid timestamp directory name: {timestamp}")

    env_dir = env_root / timestamp
    x_env_path = env_dir / "x_env.npy"
    if not x_env_path.exists():
        raise FileNotFoundError(f"Missing x_env.npy: {x_env_path}")
    x_env = np.load(x_env_path).astype(np.float32)
    if x_env.ndim != 3:
        raise ValueError(f"Expected x_env to be 3D (C,H,W): {x_env_path}")

    channels = _read_env_channel_names(env_dir, expected_channels=int(x_env.shape[0]))
    h, w = int(x_env.shape[1]), int(x_env.shape[2])

    x_bathy_path = env_dir / "x_bathy.npy"
    has_bathy = x_bathy_path.exists()
    if has_bathy:
        bathy = np.load(x_bathy_path).astype(np.float32)
        if bathy.shape != (h, w):
            raise ValueError(
                f"Bathy shape mismatch at {x_bathy_path}: got {bathy.shape}, expected {(h, w)}"
            )
    else:
        bathy = np.full((h, w), bathy_fill_value, dtype=np.float32)

    heatmap_path = heatmap_root / f"{timestamp}.npy"
    if not heatmap_path.exists():
        raise FileNotFoundError(f"Missing AIS heatmap file: {heatmap_path}")
    heatmap = np.load(heatmap_path).astype(np.float32)
    if heatmap.shape != (h, w):
        raise ValueError(
            f"Heatmap shape mismatch at {heatmap_path}: got {heatmap.shape}, expected {(h, w)}"
        )

    stack = np.concatenate([x_env, bathy[None, ...], heatmap[None, ...]], axis=0)
    channel_names = channels + ["bathy", "ais_heatmap"]
    return FeatureBundle(stack=stack, channel_names=channel_names, has_bathy=has_bathy)


def make_blocked_mask_from_bathy(
    bathy: np.ndarray,
    blocked_if_bathy_gte: float = 0.0,
) -> np.ndarray:
    if bathy.ndim != 2:
        raise ValueError("bathy must be a 2D array")
    blocked = np.zeros_like(bathy, dtype=np.uint8)
    blocked[np.isnan(bathy)] = 1
    blocked[bathy >= blocked_if_bathy_gte] = 1
    return blocked


def caution_mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("caution mask must be a 2D array")
    if mask.dtype == np.bool_:
        return mask.astype(np.uint8)
    if np.issubdtype(mask.dtype, np.integer):
        return (mask > 0).astype(np.uint8)
    return (mask >= 0.5).astype(np.uint8)


def merge_multiclass_label(
    blocked_mask: np.ndarray,
    caution_mask: np.ndarray,
) -> np.ndarray:
    if blocked_mask.shape != caution_mask.shape:
        raise ValueError(
            f"Shape mismatch blocked vs caution: {blocked_mask.shape} vs {caution_mask.shape}"
        )
    caution = caution_mask_to_uint8(caution_mask)
    blocked = caution_mask_to_uint8(blocked_mask)
    y = np.zeros(blocked.shape, dtype=np.uint8)
    y[caution > 0] = 1
    y[blocked > 0] = 2
    return y


def robust_scale_01(arr: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    vals = x[finite]
    lo = float(np.percentile(vals, p_lo))
    hi = float(np.percentile(vals, p_hi))
    if hi <= lo:
        out = np.zeros_like(x, dtype=np.float32)
        out[finite] = 0.5
        return out
    out = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32)


def quicklook_rgb(stack: np.ndarray, channel_names: list[str]) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError("stack must be 3D (C,H,W)")

    def idx_of(name: str, fallback: int) -> int:
        if name in channel_names:
            return channel_names.index(name)
        return min(fallback, stack.shape[0] - 1)

    i_ice = idx_of("ice_conc", 0)
    i_wave = idx_of("wave_hs", 2)
    i_heat = idx_of("ais_heatmap", stack.shape[0] - 1)

    r = robust_scale_01(stack[i_ice])
    g = robust_scale_01(stack[i_wave])
    b = robust_scale_01(stack[i_heat])
    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
