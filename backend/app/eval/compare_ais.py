from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.core.geo import load_grid_geo


class EvalError(RuntimeError):
    pass


def _find_heatmap(root: Path, timestamp: str) -> Path | None:
    if not root.exists():
        return None
    hits = list(root.rglob(f"{timestamp}.npy"))
    if not hits:
        return None
    return hits[0]


def _coords_from_geojson(route_geojson: dict) -> list[tuple[float, float]]:
    geometry = route_geojson.get("geometry")
    if not isinstance(geometry, dict):
        raise EvalError("route_geojson.geometry missing")
    if geometry.get("type") != "LineString":
        raise EvalError("route_geojson.geometry.type must be LineString")
    coords = geometry.get("coordinates")
    if not isinstance(coords, list) or len(coords) < 2:
        raise EvalError("route_geojson.geometry.coordinates must contain at least 2 points")
    out: list[tuple[float, float]] = []
    for item in coords:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        out.append((float(item[0]), float(item[1])))
    if len(out) < 2:
        raise EvalError("route coordinates invalid")
    return out


def evaluate_route_vs_ais_heatmap(*, settings: Settings, timestamp: str, route_geojson: dict) -> dict:
    heatmap_path = _find_heatmap(settings.ais_heatmap_root, timestamp)
    if heatmap_path is None:
        raise EvalError(f"AIS heatmap not found for timestamp={timestamp}")
    heat = np.load(heatmap_path).astype(np.float32)
    if heat.ndim != 2:
        raise EvalError(f"Invalid AIS heatmap shape: {heat.shape}")

    coords = _coords_from_geojson(route_geojson)
    h, w = heat.shape
    geo = load_grid_geo(settings, timestamp=timestamp, shape=(h, w))
    values: list[float] = []
    inside_count = 0
    for lon, lat in coords:
        r, c, inside = geo.latlon_to_rc(lat, lon)
        if inside:
            inside_count += 1
        values.append(float(heat[r, c]))
    vals = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(heat)
    global_vals = heat[finite]
    if global_vals.size == 0:
        raise EvalError("AIS heatmap has no finite values")

    p90 = float(np.percentile(global_vals, 90))
    p75 = float(np.percentile(global_vals, 75))
    p50 = float(np.percentile(global_vals, 50))
    vmax = float(np.max(global_vals))
    vmean = float(np.mean(global_vals))
    vstd = float(np.std(global_vals) + 1e-6)

    top10_hit = float(np.mean(vals >= p90))
    top25_hit = float(np.mean(vals >= p75))
    median_or_higher_hit = float(np.mean(vals >= p50))
    mean_score = float(np.mean(vals))
    p90_route = float(np.percentile(vals, 90))
    zscore = float((mean_score - vmean) / vstd)
    norm_alignment = float(np.clip(mean_score / max(vmax, 1e-6), 0.0, 1.0))

    return {
        "timestamp": timestamp,
        "source": str(heatmap_path),
        "route_point_count": int(vals.size),
        "route_inside_grid_ratio": round(inside_count / max(1, len(coords)), 4),
        "route_mean_heat": round(mean_score, 6),
        "route_p90_heat": round(p90_route, 6),
        "global_mean_heat": round(vmean, 6),
        "global_p90_heat": round(p90, 6),
        "top10pct_hit_rate": round(top10_hit, 4),
        "top25pct_hit_rate": round(top25_hit, 4),
        "median_or_higher_hit_rate": round(median_or_higher_hit, 4),
        "alignment_norm_0_1": round(norm_alignment, 4),
        "alignment_zscore": round(zscore, 4),
    }
