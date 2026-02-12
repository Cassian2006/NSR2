from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings, get_settings
from app.core.dataset import normalize_timestamp
from app.core.geo import load_grid_geo
from app.preprocess.unet_dataset import merge_multiclass_label


class AnnotationWorkspaceError(RuntimeError):
    pass


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AnnotationPoint:
    lat: float
    lon: float


@dataclass(frozen=True)
class AnnotationOperation:
    op_id: str
    mode: str  # add | erase
    shape: str  # polygon | stroke
    radius_cells: int
    points: list[AnnotationPoint]


def _polygon_mask(shape: tuple[int, int], rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    if rows.size < 3 or cols.size < 3:
        return np.zeros((h, w), dtype=bool)

    r_min = int(max(0, min(h - 1, np.floor(rows.min()) - 1)))
    r_max = int(max(0, min(h - 1, np.ceil(rows.max()) + 1)))
    c_min = int(max(0, min(w - 1, np.floor(cols.min()) - 1)))
    c_max = int(max(0, min(w - 1, np.ceil(cols.max()) + 1)))
    if r_min > r_max or c_min > c_max:
        return np.zeros((h, w), dtype=bool)

    y = (np.arange(r_min, r_max + 1, dtype=np.float64)[:, None] + 0.5)
    x = (np.arange(c_min, c_max + 1, dtype=np.float64)[None, :] + 0.5)
    inside = np.zeros((r_max - r_min + 1, c_max - c_min + 1), dtype=bool)

    x_prev = cols[-1]
    y_prev = rows[-1]
    eps = 1e-9
    for x_curr, y_curr in zip(cols, rows, strict=False):
        cond = (y_curr > y) != (y_prev > y)
        x_intersect = (x_prev - x_curr) * (y - y_curr) / ((y_prev - y_curr) + eps) + x_curr
        inside ^= cond & (x < x_intersect)
        x_prev = x_curr
        y_prev = y_curr

    # Keep polygon edges/vertices visible after rasterization.
    edge = np.zeros_like(inside)
    x_prev = cols[-1]
    y_prev = rows[-1]
    for x_curr, y_curr in zip(cols, rows, strict=False):
        n = int(max(abs(x_curr - x_prev), abs(y_curr - y_prev))) + 1
        rr = np.rint(np.linspace(y_prev, y_curr, n)).astype(np.int64)
        cc = np.rint(np.linspace(x_prev, x_curr, n)).astype(np.int64)
        rr = np.clip(rr, r_min, r_max)
        cc = np.clip(cc, c_min, c_max)
        edge[rr - r_min, cc - c_min] = True
        x_prev = x_curr
        y_prev = y_curr

    mask = np.zeros((h, w), dtype=bool)
    mask[r_min : r_max + 1, c_min : c_max + 1] = inside | edge
    return mask


def _stroke_mask(shape: tuple[int, int], rows: np.ndarray, cols: np.ndarray, radius_cells: int) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    if rows.size < 2 or cols.size < 2:
        return np.zeros((h, w), dtype=bool)
    radius_cells = int(max(1, radius_cells))
    yy, xx = np.mgrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
    disk = (yy * yy + xx * xx) <= (radius_cells * radius_cells)
    off_r = yy[disk].astype(np.int64)
    off_c = xx[disk].astype(np.int64)

    out = np.zeros((h, w), dtype=bool)
    for i in range(rows.size - 1):
        r0, c0 = float(rows[i]), float(cols[i])
        r1, c1 = float(rows[i + 1]), float(cols[i + 1])
        n = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
        rr = np.rint(np.linspace(r0, r1, n)).astype(np.int64)
        cc = np.rint(np.linspace(c0, c1, n)).astype(np.int64)
        for r, c in zip(rr, cc, strict=False):
            pr = r + off_r
            pc = c + off_c
            valid = (pr >= 0) & (pr < h) & (pc >= 0) & (pc < w)
            out[pr[valid], pc[valid]] = True
    return out


class AnnotationWorkspaceService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.root = self.settings.outputs_root / "annotation_workspace"
        self.patch_root = self.root / "patches"
        self.patch_root.mkdir(parents=True, exist_ok=True)

    def _annotation_dir(self, timestamp: str) -> Path:
        return self.settings.annotation_pack_root / timestamp

    def _patch_path(self, timestamp: str) -> Path:
        return self.patch_root / f"{timestamp}.json"

    def _load_blocked_mask(self, timestamp: str) -> np.ndarray:
        ann_dir = self._annotation_dir(timestamp)
        blocked_path = ann_dir / "blocked_mask.npy"
        if not blocked_path.exists():
            raise AnnotationWorkspaceError(f"blocked_mask missing for timestamp={timestamp}")
        blocked = np.load(blocked_path).astype(np.uint8)
        if blocked.ndim != 2:
            raise AnnotationWorkspaceError(f"blocked_mask must be 2D for timestamp={timestamp}")
        return blocked

    def _load_caution_mask(self, ann_dir: Path, shape: tuple[int, int]) -> np.ndarray:
        caution_path = ann_dir / "caution_mask.npy"
        if caution_path.exists():
            arr = np.load(caution_path)
            if arr.ndim != 2 or arr.shape != shape:
                raise AnnotationWorkspaceError(
                    f"caution_mask shape mismatch for {ann_dir.name}: got={arr.shape} expected={shape}"
                )
            return (arr > 0).astype(np.uint8)
        return np.zeros(shape, dtype=np.uint8)

    def _load_patch_operations(self, timestamp: str) -> list[dict[str, Any]]:
        path = self._patch_path(timestamp)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        operations = payload.get("operations", [])
        if not isinstance(operations, list):
            return []
        out: list[dict[str, Any]] = []
        for item in operations:
            if not isinstance(item, dict):
                continue
            mode = str(item.get("mode", "")).strip().lower()
            shape = str(item.get("shape", "polygon")).strip().lower() or "polygon"
            radius_cells = int(item.get("radius_cells", 2) or 2)
            points = item.get("points", [])
            if mode not in {"add", "erase"} or shape not in {"polygon", "stroke"} or not isinstance(points, list):
                continue
            norm_points: list[dict[str, float]] = []
            for p in points:
                if not isinstance(p, dict):
                    continue
                try:
                    lat = float(p.get("lat"))
                    lon = float(p.get("lon"))
                except Exception:
                    continue
                norm_points.append({"lat": lat, "lon": lon})
            min_points = 3 if shape == "polygon" else 2
            if len(norm_points) < min_points:
                continue
            out.append(
                {
                    "id": str(item.get("id", "")),
                    "mode": mode,
                    "shape": shape,
                    "radius_cells": int(max(1, min(40, radius_cells))),
                    "points": norm_points,
                }
            )
        return out

    def get_patch(self, timestamp_raw: str) -> dict[str, Any]:
        timestamp = normalize_timestamp(timestamp_raw)
        blocked = self._load_blocked_mask(timestamp)
        ann_dir = self._annotation_dir(timestamp)
        caution = self._load_caution_mask(ann_dir, blocked.shape)
        patch_path = self._patch_path(timestamp)
        operations = self._load_patch_operations(timestamp)

        return {
            "timestamp": timestamp,
            "operations": operations,
            "patch_file": str(patch_path),
            "caution_file": str(ann_dir / "caution_mask.npy"),
            "y_class_file": str(ann_dir / "y_class.npy"),
            "stats": {
                "shape": [int(blocked.shape[0]), int(blocked.shape[1])],
                "blocked_pixels": int((blocked > 0).sum()),
                "caution_pixels": int((caution > 0).sum()),
                "caution_ratio": float((caution > 0).sum() / max(1, caution.size)),
                "operations_count": len(operations),
                "has_patch_file": patch_path.exists(),
            },
        }

    def save_patch(
        self,
        *,
        timestamp_raw: str,
        operations_raw: list[dict[str, Any]],
        note: str = "",
        author: str = "web",
    ) -> dict[str, Any]:
        timestamp = normalize_timestamp(timestamp_raw)
        if len(operations_raw) > 5000:
            raise AnnotationWorkspaceError("too many operations; max=5000")

        blocked = self._load_blocked_mask(timestamp)
        ann_dir = self._annotation_dir(timestamp)
        ann_dir.mkdir(parents=True, exist_ok=True)
        caution = self._load_caution_mask(ann_dir, blocked.shape)
        geo = load_grid_geo(self.settings, timestamp=timestamp, shape=blocked.shape)

        operations_norm: list[dict[str, Any]] = []
        for idx, raw in enumerate(operations_raw, start=1):
            if not isinstance(raw, dict):
                raise AnnotationWorkspaceError(f"operation[{idx}] must be an object")
            mode = str(raw.get("mode", "")).strip().lower()
            if mode not in {"add", "erase"}:
                raise AnnotationWorkspaceError(f"operation[{idx}].mode must be add/erase")
            shape = str(raw.get("shape", "polygon")).strip().lower() or "polygon"
            if shape not in {"polygon", "stroke"}:
                raise AnnotationWorkspaceError(f"operation[{idx}].shape must be polygon/stroke")
            try:
                radius_cells = int(raw.get("radius_cells", 2) or 2)
            except Exception as exc:
                raise AnnotationWorkspaceError(f"operation[{idx}].radius_cells invalid") from exc
            radius_cells = int(max(1, min(40, radius_cells)))
            raw_points = raw.get("points", [])
            min_points = 3 if shape == "polygon" else 2
            if not isinstance(raw_points, list) or len(raw_points) < min_points:
                raise AnnotationWorkspaceError(f"operation[{idx}] requires at least {min_points} points")
            points: list[AnnotationPoint] = []
            for p_idx, p in enumerate(raw_points, start=1):
                if not isinstance(p, dict):
                    raise AnnotationWorkspaceError(f"operation[{idx}].points[{p_idx}] must be an object")
                try:
                    lat = float(p.get("lat"))
                    lon = float(p.get("lon"))
                except Exception as exc:
                    raise AnnotationWorkspaceError(
                        f"operation[{idx}].points[{p_idx}] lat/lon invalid"
                    ) from exc
                points.append(AnnotationPoint(lat=lat, lon=lon))

            lats = np.asarray([p.lat for p in points], dtype=np.float64)
            lons = np.asarray([p.lon for p in points], dtype=np.float64)
            rows = geo.row_coords_for_lats(lats)
            cols = geo.col_coords_for_lons(lons)
            if shape == "polygon":
                mask = _polygon_mask(blocked.shape, rows=rows, cols=cols)
            else:
                mask = _stroke_mask(blocked.shape, rows=rows, cols=cols, radius_cells=radius_cells)
            if mode == "add":
                caution[mask & (blocked == 0)] = 1
            else:
                caution[mask] = 0

            operations_norm.append(
                {
                    "id": str(raw.get("id", "")) or f"op_{idx:04d}",
                    "mode": mode,
                    "shape": shape,
                    "radius_cells": radius_cells,
                    "points": [{"lat": float(p.lat), "lon": float(p.lon)} for p in points],
                }
            )

        caution[blocked > 0] = 0
        y_class = merge_multiclass_label(blocked_mask=blocked, caution_mask=caution)

        caution_path = ann_dir / "caution_mask.npy"
        y_class_path = ann_dir / "y_class.npy"
        np.save(caution_path, caution.astype(np.uint8))
        np.save(y_class_path, y_class.astype(np.uint8))

        patch_payload = {
            "timestamp": timestamp,
            "updated_at": _utc_now_iso(),
            "author": author,
            "note": note.strip(),
            "operations": operations_norm,
            "stats": {
                "shape": [int(blocked.shape[0]), int(blocked.shape[1])],
                "blocked_pixels": int((blocked > 0).sum()),
                "caution_pixels": int((caution > 0).sum()),
                "caution_ratio": float((caution > 0).sum() / max(1, caution.size)),
                "operations_count": len(operations_norm),
            },
            "outputs": {
                "caution_file": str(caution_path),
                "y_class_file": str(y_class_path),
            },
        }
        patch_path = self._patch_path(timestamp)
        patch_path.write_text(json.dumps(patch_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "timestamp": timestamp,
            "updated_at": patch_payload["updated_at"],
            "patch_file": str(patch_path),
            "caution_file": str(caution_path),
            "y_class_file": str(y_class_path),
            "stats": patch_payload["stats"],
            "operations": operations_norm,
        }
