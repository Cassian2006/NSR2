from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings


@dataclass(frozen=True)
class ShapeInfo:
    layer: str
    present: bool
    shape: tuple[int, int] | None
    aligned: bool | None

    def to_json(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "present": self.present,
            "shape": list(self.shape) if self.shape is not None else None,
            "aligned": self.aligned,
        }


def _load_hw(path: Path, *, is_stack: bool) -> tuple[int, int] | None:
    if not path.exists():
        return None
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception:
        return None
    if is_stack:
        if arr.ndim != 3:
            return None
        return int(arr.shape[1]), int(arr.shape[2])
    if arr.ndim != 2:
        return None
    return int(arr.shape[0]), int(arr.shape[1])


def _find_ais_hw(ais_root: Path, timestamp: str) -> tuple[int, int] | None:
    for p in sorted(ais_root.rglob(f"{timestamp}.npy")):
        return _load_hw(p, is_stack=False)
    return None


def _scan_source_timestamps(annotation_pack_root: Path) -> list[str]:
    if not annotation_pack_root.exists():
        return []
    out: list[str] = []
    for folder in sorted(annotation_pack_root.iterdir()):
        if not folder.is_dir():
            continue
        ts = folder.name
        if len(ts) == 13 and ts[4] == "-" and ts[7] == "-" and ts[10] == "_":
            out.append(ts)
    return out


def _pick_samples(values: list[str], sample_limit: int) -> list[str]:
    if len(values) <= sample_limit:
        return values
    idx = np.linspace(0, len(values) - 1, num=sample_limit, dtype=np.int64)
    return [values[int(i)] for i in idx.tolist()]


def _majority_shape(shapes: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not shapes:
        return None
    counts: dict[tuple[int, int], int] = {}
    for shp in shapes:
        counts[shp] = counts.get(shp, 0) + 1
    return max(counts, key=lambda k: counts[k])


def _timestamp_layer_shapes(
    *,
    annotation_pack_root: Path,
    ais_heatmap_root: Path,
    pred_root: Path,
    timestamp: str,
) -> dict[str, tuple[int, int] | None]:
    ann = annotation_pack_root / timestamp
    pred = pred_root / "unet_v1"
    return {
        "env_stack": _load_hw(ann / "x_stack.npy", is_stack=True),
        "bathy": _load_hw(ann / "blocked_mask.npy", is_stack=False),
        "ais_heatmap": _find_ais_hw(ais_heatmap_root, timestamp),
        "unet_pred": _load_hw(pred / f"{timestamp}.npy", is_stack=False),
        "unet_uncertainty": _load_hw(pred / f"{timestamp}_uncertainty.npy", is_stack=False),
    }


@lru_cache(maxsize=2048)
def _alignment_cached(
    annotation_pack_root: str,
    ais_heatmap_root: str,
    pred_root: str,
    timestamp: str,
) -> dict[str, Any]:
    shapes = _timestamp_layer_shapes(
        annotation_pack_root=Path(annotation_pack_root),
        ais_heatmap_root=Path(ais_heatmap_root),
        pred_root=Path(pred_root),
        timestamp=timestamp,
    )
    expected = (
        shapes.get("bathy")
        or shapes.get("env_stack")
        or shapes.get("ais_heatmap")
        or shapes.get("unet_pred")
        or shapes.get("unet_uncertainty")
    )

    items: list[ShapeInfo] = []
    for layer, shp in shapes.items():
        if shp is None:
            items.append(ShapeInfo(layer=layer, present=False, shape=None, aligned=None))
            continue
        aligned = True if expected is None else (shp == expected)
        items.append(ShapeInfo(layer=layer, present=True, shape=shp, aligned=aligned))

    mismatch = [i.layer for i in items if i.present and i.aligned is False]
    return {
        "timestamp": timestamp,
        "expected_shape": list(expected) if expected is not None else None,
        "ok": len(mismatch) == 0,
        "mismatch_layers": mismatch,
        "layers": [i.to_json() for i in items],
    }


def get_timestamp_alignment(settings: Settings, timestamp: str) -> dict[str, Any]:
    return _alignment_cached(
        annotation_pack_root=str(settings.annotation_pack_root),
        ais_heatmap_root=str(settings.ais_heatmap_root),
        pred_root=str(settings.pred_root),
        timestamp=timestamp,
    )


def build_grid_alignment_report(
    *,
    settings: Settings,
    sample_limit: int = 80,
) -> dict[str, Any]:
    timestamps = _scan_source_timestamps(settings.annotation_pack_root)
    sampled = _pick_samples(timestamps, max(1, int(sample_limit)))
    checks = [get_timestamp_alignment(settings, ts) for ts in sampled]
    mismatch_rows = [c for c in checks if not bool(c.get("ok", False))]

    bathy_shapes: list[tuple[int, int]] = []
    for c in checks:
        for layer in c.get("layers", []):
            if layer.get("layer") == "bathy" and layer.get("shape") is not None:
                shp = layer["shape"]
                if isinstance(shp, list) and len(shp) == 2:
                    bathy_shapes.append((int(shp[0]), int(shp[1])))
    canonical = _majority_shape(bathy_shapes)

    lat_res = None
    lon_res = None
    if canonical is not None:
        h, w = canonical
        if h > 1:
            lat_res = abs(float(settings.grid_lat_max) - float(settings.grid_lat_min)) / float(h - 1)
        if w > 1:
            lon_res = abs(float(settings.grid_lon_max) - float(settings.grid_lon_min)) / float(w - 1)

    coverage = float((len(sampled) - len(mismatch_rows)) / len(sampled)) if sampled else 0.0
    status = "pass"
    if not sampled:
        status = "fail"
    elif mismatch_rows:
        status = "warn"
    if canonical is None:
        status = "fail"

    grid_spec = {
        "crs": {
            "compute": "EPSG:3413",
            "display": "EPSG:4326",
            "web_map": "EPSG:3857",
            "strategy": "Compute on polar stereographic grid and display in WGS84/WebMercator.",
        },
        "grid": {
            "shape": list(canonical) if canonical is not None else None,
            "bounds": {
                "lat_min": float(settings.grid_lat_min),
                "lat_max": float(settings.grid_lat_max),
                "lon_min": float(settings.grid_lon_min),
                "lon_max": float(settings.grid_lon_max),
            },
            "resolution": {
                "lat_deg": lat_res,
                "lon_deg": lon_res,
            },
        },
        "alignment": {
            "checked_timestamps": len(sampled),
            "mismatch_timestamps": len(mismatch_rows),
            "coverage": round(coverage, 6),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "summary": {
            "status": status,
            "timestamp_count": len(timestamps),
            "sampled_count": len(sampled),
            "mismatch_count": len(mismatch_rows),
            "coverage": round(coverage, 6),
        },
        "grid_spec": grid_spec,
        "mismatches": mismatch_rows[:200],
    }
