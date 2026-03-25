from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np


def _grow_mask_once(mask: np.ndarray) -> np.ndarray:
    pad = np.pad(mask, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    return (
        pad[1:-1, 1:-1]
        | pad[:-2, 1:-1]
        | pad[2:, 1:-1]
        | pad[1:-1, :-2]
        | pad[1:-1, 2:]
        | pad[:-2, :-2]
        | pad[:-2, 2:]
        | pad[2:, :-2]
        | pad[2:, 2:]
    )


def _box_blur3x3(field: np.ndarray, passes: int = 1) -> np.ndarray:
    out = field.astype(np.float32)
    for _ in range(max(0, int(passes))):
        pad = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            pad[:-2, :-2]
            + pad[:-2, 1:-1]
            + pad[:-2, 2:]
            + pad[1:-1, :-2]
            + pad[1:-1, 1:-1]
            + pad[1:-1, 2:]
            + pad[2:, :-2]
            + pad[2:, 1:-1]
            + pad[2:, 2:]
        ) / 9.0
    return out.astype(np.float32)


def _normalize01(arr: np.ndarray) -> np.ndarray:
    out = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vmax = float(np.max(out)) if out.size else 0.0
    if vmax > 1e-6:
        out /= vmax
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _iter_heatmap_files(ais_root: Path) -> list[Path]:
    if not ais_root.exists():
        return []
    return sorted(p for p in ais_root.rglob("*.npy") if p.is_file())


def _iter_cleaned_csvs(cleaned_root: Path) -> list[Path]:
    if not cleaned_root.exists():
        return []
    return sorted(p for p in cleaned_root.rglob("*_clean.csv") if p.is_file())


def _cache_path(ais_root: Path, name: str, shape: tuple[int, int]) -> Path:
    cache_dir = ais_root / "_display_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}_{shape[0]}x{shape[1]}.npy"


@lru_cache(maxsize=8)
def _corridor_prior_cached(ais_root_str: str, shape: tuple[int, int]) -> np.ndarray | None:
    ais_root = Path(ais_root_str)
    files = _iter_heatmap_files(ais_root)
    if not files:
        return None

    stack_max = np.zeros(shape, dtype=np.float32)
    matched = 0
    for path in files:
        try:
            arr = np.load(path).astype(np.float32)
        except Exception:
            continue
        if arr.ndim != 2 or tuple(arr.shape) != shape:
            continue
        stack_max = np.maximum(stack_max, np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
        matched += 1

    if matched == 0 or float(np.max(stack_max)) <= 1e-6:
        return None

    cache_path = _cache_path(ais_root, "heat_prior_v2", shape)
    if cache_path.exists():
        try:
            cached = np.load(cache_path).astype(np.float32)
            if cached.shape == shape:
                return cached
        except Exception:
            pass

    smooth = _normalize01(_box_blur3x3(stack_max, passes=4))
    grown = stack_max > 1e-6
    for _ in range(18):
        grown = _grow_mask_once(grown)
    corridor = _normalize01(_box_blur3x3(grown.astype(np.float32), passes=4))
    prior = _normalize01(smooth * 0.88 + corridor * 0.12)
    try:
        np.save(cache_path, prior.astype(np.float32))
    except Exception:
        pass
    return prior


def _to_cell(lat: float, lon: float, shape: tuple[int, int]) -> tuple[int, int] | None:
    lat_min, lat_max = 60.0, 80.0
    lon_min, lon_max = 20.0, 180.0
    if lat < lat_min or lat > lat_max or lon < lon_min or lon > lon_max:
        return None
    h, w = shape
    row = int((lat_max - lat) / max(1e-6, (lat_max - lat_min)) * h)
    col = int((lon - lon_min) / max(1e-6, (lon_max - lon_min)) * w)
    row = min(max(row, 0), h - 1)
    col = min(max(col, 0), w - 1)
    return row, col


@lru_cache(maxsize=8)
def _cleaned_prior_cached(cleaned_root_str: str, shape: tuple[int, int]) -> np.ndarray | None:
    cleaned_root = Path(cleaned_root_str)
    ais_root = cleaned_root.parents[1] / "ais_heatmap"
    cache_path = _cache_path(ais_root, "cleaned_prior_v2", shape)
    if cache_path.exists():
        try:
            cached = np.load(cache_path).astype(np.float32)
            if cached.shape == shape:
                return cached
        except Exception:
            pass
    files = _iter_cleaned_csvs(cleaned_root)
    if not files:
        return None

    hist = np.zeros(shape, dtype=np.float32)
    matched = 0
    for path in files:
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        lat = float(row.get("lat", "nan"))
                        lon = float(row.get("lon", "nan"))
                    except (TypeError, ValueError):
                        continue
                    cell = _to_cell(lat, lon, shape)
                    if cell is None:
                        continue
                    hist[cell] += 1.0
                    matched += 1
        except Exception:
            continue

    if matched == 0 or float(np.max(hist)) <= 1e-6:
        return None

    hist = _normalize01(np.log1p(hist))
    smooth = _normalize01(_box_blur3x3(hist, passes=5))
    grown = hist > 1e-6
    for _ in range(22):
        grown = _grow_mask_once(grown)
    corridor = _normalize01(_box_blur3x3(grown.astype(np.float32), passes=4))
    prior = _normalize01(smooth * 0.82 + corridor * 0.18)
    try:
        np.save(cache_path, prior.astype(np.float32))
    except Exception:
        pass
    return prior


def load_ais_corridor_prior(ais_root: Path, shape: tuple[int, int]) -> np.ndarray | None:
    return _corridor_prior_cached(str(ais_root.resolve()), tuple(int(v) for v in shape))


def build_ais_display_grid(
    *,
    ais_root: Path,
    cleaned_root: Path | None = None,
    local_grid: np.ndarray | None = None,
    shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    target_shape = tuple(local_grid.shape) if local_grid is not None else shape
    if target_shape is None:
        files = _iter_heatmap_files(ais_root)
        for path in files:
            try:
                arr = np.load(path, mmap_mode="r")
            except Exception:
                continue
            if arr.ndim == 2:
                target_shape = tuple(int(v) for v in arr.shape)
                break
    if target_shape is None:
        return None

    parts: list[np.ndarray] = []
    heatmap_prior = load_ais_corridor_prior(ais_root, target_shape)
    if heatmap_prior is not None:
        parts.append(heatmap_prior * 0.18)
    if cleaned_root is not None:
        cleaned_prior = _cleaned_prior_cached(str(cleaned_root.resolve()), target_shape)
        if cleaned_prior is not None:
            parts.append(cleaned_prior)
    prior = _normalize01(np.maximum.reduce(parts)) if parts else None
    if local_grid is None:
        return prior

    local = _normalize01(local_grid)
    if prior is None:
        return local
    blended = np.maximum(local, local * 0.72 + prior * 0.28)
    return _normalize01(blended)
