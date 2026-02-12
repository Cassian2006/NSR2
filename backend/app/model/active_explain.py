from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_CHANNELS = ["ice_conc", "ice_thick", "wave_hs", "wind_u10", "wind_v10", "bathy", "ais_heatmap"]


def load_channel_names(sample_dir: Path, n_channels: int) -> list[str]:
    meta_path = sample_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            names = meta.get("channel_names")
            if isinstance(names, list) and len(names) == n_channels:
                return [str(v) for v in names]
        except Exception:
            pass
    if n_channels == len(DEFAULT_CHANNELS):
        return list(DEFAULT_CHANNELS)
    return [f"ch_{i:02d}" for i in range(n_channels)]


def _channel_index(channel_names: list[str], keys: list[str], fallback: int) -> tuple[int, str]:
    lowered = [c.lower() for c in channel_names]
    for key in keys:
        for i, ch in enumerate(lowered):
            if key in ch:
                return i, channel_names[i]
    idx = max(0, min(int(fallback), len(channel_names) - 1))
    return idx, channel_names[idx]


def _normalize_on_sea(arr: np.ndarray, sea: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    if arr.shape != sea.shape:
        return out
    vals = np.asarray(arr[sea], dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return out
    lo = float(np.percentile(vals, 5))
    hi = float(np.percentile(vals, 95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return out
    out = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def _dilated_boundary_mask(blocked: np.ndarray, iters: int = 2) -> np.ndarray:
    edge = np.zeros_like(blocked, dtype=bool)
    edge[1:, :] |= blocked[1:, :] != blocked[:-1, :]
    edge[:-1, :] |= blocked[:-1, :] != blocked[1:, :]
    edge[:, 1:] |= blocked[:, 1:] != blocked[:, :-1]
    edge[:, :-1] |= blocked[:, :-1] != blocked[:, 1:]
    near = edge.copy()
    for _ in range(max(0, iters)):
        n = near.copy()
        n[1:, :] |= near[:-1, :]
        n[:-1, :] |= near[1:, :]
        n[:, 1:] |= near[:, :-1]
        n[:, :-1] |= near[:, 1:]
        near = n
    return near


def explain_sample(
    *,
    x_stack: np.ndarray,
    blocked_mask: np.ndarray,
    caution_prob: np.ndarray,
    entropy: np.ndarray,
    channel_names: list[str],
    suggested_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if x_stack.ndim != 3:
        raise ValueError("x_stack must be 3D (C,H,W)")
    h, w = x_stack.shape[1], x_stack.shape[2]
    if blocked_mask.shape != (h, w) or caution_prob.shape != (h, w) or entropy.shape != (h, w):
        raise ValueError("blocked/caution_prob/entropy shape mismatch")

    sea = blocked_mask == 0
    sea_count = int(sea.sum())
    if sea_count <= 0:
        raise ValueError("no sea pixels found in sample")

    i_ice, ch_ice = _channel_index(channel_names, ["ice_conc", "ice"], fallback=0)
    i_wave, ch_wave = _channel_index(channel_names, ["wave_hs", "wave"], fallback=min(2, len(channel_names) - 1))
    i_ais, ch_ais = _channel_index(channel_names, ["ais_heatmap", "ais"], fallback=len(channel_names) - 1)

    wind_u_idx, ch_wu = _channel_index(channel_names, ["wind_u10", "u10", "wind_u"], fallback=min(3, len(channel_names) - 1))
    wind_v_idx, ch_wv = _channel_index(channel_names, ["wind_v10", "v10", "wind_v"], fallback=min(4, len(channel_names) - 1))

    ice = _normalize_on_sea(x_stack[i_ice], sea)
    wave = _normalize_on_sea(x_stack[i_wave], sea)
    ais = _normalize_on_sea(x_stack[i_ais], sea)
    wind_mag = np.sqrt(np.square(x_stack[wind_u_idx].astype(np.float32)) + np.square(x_stack[wind_v_idx].astype(np.float32)))
    wind = _normalize_on_sea(wind_mag, sea)

    cp = np.clip(caution_prob.astype(np.float32), 0.0, 1.0)
    ent = np.clip(entropy.astype(np.float32), 0.0, None)
    boundary = _dilated_boundary_mask(blocked_mask, iters=2) & sea
    boundary_unc = float(ent[boundary].mean()) if int(boundary.sum()) > 0 else float(ent[sea].mean())

    raw = {
        "ice_contribution": float((cp[sea] * ice[sea]).mean()),
        "wave_contribution": float((cp[sea] * wave[sea]).mean()),
        "wind_contribution": float((cp[sea] * wind[sea]).mean()),
        "ais_deviation": float((cp[sea] * (1.0 - ais[sea])).mean()),
        # proxy risk: high uncertainty + near-boundary uncertainty usually correlates with likely misclassification regions
        "historical_misclassification_risk": float(0.6 * ent[sea].mean() + 0.4 * boundary_unc),
    }

    raw = {k: max(0.0, float(v)) for k, v in raw.items()}
    total_raw = float(sum(raw.values()))
    if total_raw > 0.0:
        norm = {k: float(v / total_raw) for k, v in raw.items()}
    else:
        n = float(len(raw))
        norm = {k: float(1.0 / n) for k in raw}
    dominant = max(norm.items(), key=lambda kv: kv[1])[0]

    suggested_ratio = 0.0
    if suggested_mask is not None and suggested_mask.shape == (h, w):
        suggested_ratio = float((suggested_mask[sea] > 0).mean())

    return {
        "version": "v1",
        "factors_raw": raw,
        "factors_norm": norm,
        "dominant_factor": dominant,
        "total_raw": total_raw,
        "stats": {
            "sea_pixels": sea_count,
            "entropy_mean": float(ent[sea].mean()),
            "entropy_p95": float(np.percentile(ent[sea], 95)),
            "boundary_uncertainty_mean": boundary_unc,
            "suggested_ratio": suggested_ratio,
        },
        "channels_used": {
            "ice": ch_ice,
            "wave": ch_wave,
            "wind_u": ch_wu,
            "wind_v": ch_wv,
            "ais": ch_ais,
        },
        "notes": {
            "historical_misclassification_risk": "proxy_from_model_uncertainty_and_boundary_context",
        },
    }


def render_explanation_card(explain: dict[str, Any], out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    w, h = 920, 420
    img = Image.new("RGB", (w, h), (247, 249, 252))
    draw = ImageDraw.Draw(img)

    draw.text((24, 18), f"Explanation Snapshot - {title}", fill=(24, 33, 55))
    draw.text((24, 44), "Normalized factor contributions (sum=1)", fill=(85, 97, 121))

    factors = explain.get("factors_norm", {})
    items = [
        ("ice_contribution", (66, 133, 244)),
        ("wave_contribution", (52, 168, 83)),
        ("wind_contribution", (251, 188, 5)),
        ("ais_deviation", (234, 67, 53)),
        ("historical_misclassification_risk", (123, 31, 162)),
    ]
    x0 = 30
    y0 = 90
    bar_w = 640
    bar_h = 34
    gap = 18
    for idx, (name, color) in enumerate(items):
        y = y0 + idx * (bar_h + gap)
        v = float(factors.get(name, 0.0))
        draw.rectangle((x0, y, x0 + bar_w, y + bar_h), outline=(210, 217, 228), width=1, fill=(255, 255, 255))
        draw.rectangle((x0, y, x0 + int(bar_w * max(0.0, min(1.0, v))), y + bar_h), fill=color)
        draw.text((x0 + bar_w + 18, y + 8), f"{v:.3f}", fill=(33, 44, 68))
        draw.text((x0 + 8, y + 8), name, fill=(255, 255, 255) if v > 0.18 else (33, 44, 68))

    stats = explain.get("stats", {})
    draw.text((24, 340), f"dominant_factor: {explain.get('dominant_factor', '')}", fill=(24, 33, 55))
    draw.text((24, 364), f"entropy_mean={float(stats.get('entropy_mean', 0.0)):.4f}", fill=(24, 33, 55))
    draw.text((280, 364), f"entropy_p95={float(stats.get('entropy_p95', 0.0)):.4f}", fill=(24, 33, 55))
    draw.text((540, 364), f"suggested_ratio={float(stats.get('suggested_ratio', 0.0)):.4f}", fill=(24, 33, 55))

    img.save(out_png)

