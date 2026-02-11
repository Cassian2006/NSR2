from __future__ import annotations

import binascii
import json
import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.core.geo import GridGeo, load_grid_geo
from app.model.infer import InferenceError, run_unet_inference


@dataclass(frozen=True)
class GridBounds:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass(frozen=True)
class BBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


_GRID_BOUNDS = GridBounds(lat_min=60.0, lat_max=86.0, lon_min=-180.0, lon_max=180.0)


def parse_bbox(raw: str | None, *, bounds: GridBounds | None = None) -> BBox:
    b = bounds or _GRID_BOUNDS
    if not raw:
        return BBox(
            min_lon=b.lon_min,
            min_lat=b.lat_min,
            max_lon=b.lon_max,
            max_lat=b.lat_max,
        )
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'minLon,minLat,maxLon,maxLat'")
    min_lon, min_lat, max_lon, max_lat = [float(v) for v in parts]
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("bbox min values must be less than max values")
    return BBox(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)


def parse_size(raw: str | None, *, fallback_w: int = 1024, fallback_h: int = 768) -> tuple[int, int]:
    if not raw:
        return fallback_w, fallback_h
    token = raw.lower().replace("x", ",")
    parts = [p.strip() for p in token.split(",")]
    if len(parts) != 2:
        raise ValueError("size must be 'width,height' or 'widthxheight'")
    w = int(parts[0])
    h = int(parts[1])
    if w <= 0 or h <= 0:
        raise ValueError("size must be positive")
    return min(w, 4096), min(h, 4096)


def tile_bbox(z: int, x: int, y: int) -> BBox:
    n = 2**z
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0

    def _tile_y_to_lat(yy: int) -> float:
        merc = np.pi * (1.0 - 2.0 * yy / n)
        return float(np.degrees(np.arctan(np.sinh(merc))))

    lat_max = _tile_y_to_lat(y)
    lat_min = _tile_y_to_lat(y + 1)
    return BBox(min_lon=lon_min, min_lat=lat_min, max_lon=lon_max, max_lat=lat_max)


def _tile_pixel_latlon_axes(z: int, x: int, y: int, tile_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-pixel center lat/lon axes for XYZ tiles in Web Mercator.
    """
    n = 2**z
    px = (np.arange(tile_size, dtype=np.float64) + 0.5) / tile_size

    lon = ((x + px) / n) * 360.0 - 180.0

    tile_y = y + px
    merc = np.pi * (1.0 - 2.0 * (tile_y / n))
    lat = np.degrees(np.arctan(np.sinh(merc)))
    return lat, lon


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(tag)
    crc = binascii.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _encode_png_rgba(image: np.ndarray) -> bytes:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Expected uint8 RGBA image")
    h, w, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(h))
    out = bytearray()
    out.extend(b"\x89PNG\r\n\x1a\n")
    out.extend(_png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)))
    out.extend(_png_chunk(b"IDAT", zlib.compress(raw, level=6)))
    out.extend(_png_chunk(b"IEND", b""))
    return bytes(out)


def _lerp_palette(norm: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    out = np.zeros((*norm.shape, 3), dtype=np.float32)
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        if p1 <= p0:
            continue
        mask = (norm >= p0) & (norm <= p1)
        if not mask.any():
            continue
        t = (norm[mask] - p0) / (p1 - p0)
        c0v = np.asarray(c0, dtype=np.float32)
        c1v = np.asarray(c1, dtype=np.float32)
        out[mask] = c0v + (c1v - c0v) * t[:, None]
    out[norm <= stops[0][0]] = np.asarray(stops[0][1], dtype=np.float32)
    out[norm >= stops[-1][0]] = np.asarray(stops[-1][1], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _sample_grid_from_axes(
    data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    geo: GridGeo,
    mode: str = "nearest",
) -> tuple[np.ndarray, np.ndarray]:
    b = geo.bounds
    lat_in = (lats >= b.lat_min) & (lats <= b.lat_max)
    lon_in = (lons >= b.lon_min) & (lons <= b.lon_max)
    inside = lat_in[:, None] & lon_in[None, :]

    if mode == "nearest":
        row = geo.rows_for_lats(lats)
        col = geo.cols_for_lons(lons)
        sampled = data[np.ix_(row, col)]
        return sampled, inside

    if mode != "bilinear":
        raise ValueError(f"Unsupported sampling mode: {mode}")

    row_f = geo.row_coords_for_lats(lats)
    col_f = geo.col_coords_for_lons(lons)
    r0 = np.floor(row_f).astype(np.int64)
    c0 = np.floor(col_f).astype(np.int64)
    r1 = np.clip(r0 + 1, 0, data.shape[0] - 1)
    c1 = np.clip(c0 + 1, 0, data.shape[1] - 1)
    dr = (row_f - r0).astype(np.float32)
    dc = (col_f - c0).astype(np.float32)

    v00 = data[np.ix_(r0, c0)].astype(np.float32)
    v01 = data[np.ix_(r0, c1)].astype(np.float32)
    v10 = data[np.ix_(r1, c0)].astype(np.float32)
    v11 = data[np.ix_(r1, c1)].astype(np.float32)

    w00 = (1.0 - dr)[:, None] * (1.0 - dc)[None, :]
    w01 = (1.0 - dr)[:, None] * dc[None, :]
    w10 = dr[:, None] * (1.0 - dc)[None, :]
    w11 = dr[:, None] * dc[None, :]

    finite00 = np.isfinite(v00)
    finite01 = np.isfinite(v01)
    finite10 = np.isfinite(v10)
    finite11 = np.isfinite(v11)

    num = np.zeros_like(v00, dtype=np.float32)
    den = np.zeros_like(v00, dtype=np.float32)
    num += np.where(finite00, v00 * w00, 0.0)
    den += np.where(finite00, w00, 0.0)
    num += np.where(finite01, v01 * w01, 0.0)
    den += np.where(finite01, w01, 0.0)
    num += np.where(finite10, v10 * w10, 0.0)
    den += np.where(finite10, w10, 0.0)
    num += np.where(finite11, v11 * w11, 0.0)
    den += np.where(finite11, w11, 0.0)
    sampled = np.divide(num, den, out=np.full_like(num, np.nan, dtype=np.float32), where=den > 0)
    return sampled, inside


def _sample_grid(
    data: np.ndarray,
    bbox: BBox,
    out_w: int,
    out_h: int,
    *,
    geo: GridGeo,
    mode: str = "nearest",
) -> tuple[np.ndarray, np.ndarray]:
    lats = np.linspace(bbox.max_lat, bbox.min_lat, out_h, dtype=np.float64)
    lons = np.linspace(bbox.min_lon, bbox.max_lon, out_w, dtype=np.float64)
    return _sample_grid_from_axes(data, lats, lons, geo=geo, mode=mode)


def _empty_image(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


def _render_continuous(
    sampled: np.ndarray,
    inside: np.ndarray,
    *,
    stops: list[tuple[float, tuple[int, int, int]]],
    alpha_min: int,
    alpha_max: int,
    value_mask: np.ndarray | None = None,
) -> np.ndarray:
    image = _empty_image(sampled.shape[1], sampled.shape[0])
    finite = np.isfinite(sampled) & inside
    if value_mask is not None and value_mask.shape == sampled.shape:
        finite &= value_mask
    if not finite.any():
        return image

    vals = sampled[finite]
    vmin = float(np.percentile(vals, 2))
    vmax = float(np.percentile(vals, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((sampled - vmin) / (vmax - vmin), 0.0, 1.0)

    rgb = _lerp_palette(norm, stops)
    image[..., :3] = rgb

    alpha = (alpha_min + norm * (alpha_max - alpha_min)).astype(np.uint8)
    image[..., 3] = np.where(finite, alpha, 0).astype(np.uint8)
    return image


def _render_bathy(sampled: np.ndarray, inside: np.ndarray) -> np.ndarray:
    image = _empty_image(sampled.shape[1], sampled.shape[0])
    blocked = (sampled > 0.5) & inside
    image[blocked, 0] = 10
    image[blocked, 1] = 10
    image[blocked, 2] = 10
    image[blocked, 3] = 190
    return image


def _render_unet(
    sampled: np.ndarray,
    inside: np.ndarray,
    *,
    bathy_blocked_sampled: np.ndarray | None = None,
) -> np.ndarray:
    image = _empty_image(sampled.shape[1], sampled.shape[0])
    caution = (sampled == 1) & inside
    blocked = (sampled == 2) & inside
    if bathy_blocked_sampled is not None and bathy_blocked_sampled.shape == sampled.shape:
        bathy_blocked = (bathy_blocked_sampled > 0.5) & inside
        caution &= ~bathy_blocked
        blocked &= ~bathy_blocked
    image[caution] = np.asarray([245, 158, 11, 145], dtype=np.uint8)
    image[blocked] = np.asarray([239, 68, 68, 185], dtype=np.uint8)
    return image


def _normalize_ice_sampled(sampled: np.ndarray) -> np.ndarray:
    # In source grids, open water is often represented as NaN for ice_conc.
    # Treat it as 0% concentration so the ice layer stays spatially continuous over sea.
    out = np.nan_to_num(sampled.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out, 0.0, 100.0)


@lru_cache(maxsize=4)
def _load_pack(timestamp: str, annotation_pack_root: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pack_dir = Path(annotation_pack_root) / timestamp
    x_stack = np.load(pack_dir / "x_stack.npy").astype(np.float32)
    blocked = np.load(pack_dir / "blocked_mask.npy").astype(np.float32)
    meta_path = pack_dir / "meta.json"
    channel_names: list[str] = []
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        channel_names = list(meta.get("channel_names", []))
    return x_stack, blocked, channel_names


def _channel_idx(names: list[str], key: str) -> int | None:
    try:
        return names.index(key)
    except ValueError:
        return None


def _load_layer_grid(settings: Settings, timestamp: str, layer: str) -> np.ndarray | None:
    try:
        x_stack, blocked, channels = _load_pack(timestamp, str(settings.annotation_pack_root))
    except Exception:
        x_stack, blocked, channels = None, None, []

    if layer == "bathy":
        return blocked

    if layer == "ais_heatmap":
        if x_stack is not None:
            idx = _channel_idx(channels, "ais_heatmap")
            if idx is not None:
                return x_stack[idx]
        hits = list(settings.ais_heatmap_root.rglob(f"{timestamp}.npy"))
        if hits:
            return np.load(hits[0]).astype(np.float32)
        return None

    if layer == "unet_pred":
        pred_path = settings.pred_root / "unet_v1" / f"{timestamp}.npy"
        if not pred_path.exists():
            try:
                run_unet_inference(
                    settings=settings,
                    timestamp=timestamp,
                    model_version="unet_v1",
                    output_path=pred_path,
                )
            except InferenceError:
                return None
        if pred_path.exists():
            return np.load(pred_path).astype(np.float32)
        return None

    if layer == "unet_uncertainty":
        unc_path = settings.pred_root / "unet_v1" / f"{timestamp}_uncertainty.npy"
        if not unc_path.exists():
            pred_path = settings.pred_root / "unet_v1" / f"{timestamp}.npy"
            try:
                run_unet_inference(
                    settings=settings,
                    timestamp=timestamp,
                    model_version="unet_v1",
                    output_path=pred_path,
                )
            except InferenceError:
                return None
        if unc_path.exists():
            return np.load(unc_path).astype(np.float32)
        return None

    if x_stack is None:
        return None

    if layer == "ice":
        idx = _channel_idx(channels, "ice_conc")
        return x_stack[idx] if idx is not None else None
    if layer == "wave":
        idx = _channel_idx(channels, "wave_hs")
        return x_stack[idx] if idx is not None else None
    if layer == "wind":
        iu = _channel_idx(channels, "wind_u10")
        iv = _channel_idx(channels, "wind_v10")
        if iu is None or iv is None:
            return None
        return np.sqrt(x_stack[iu] ** 2 + x_stack[iv] ** 2)
    return None


def render_overlay_png(
    *,
    settings: Settings,
    timestamp: str,
    layer: str,
    bbox: BBox,
    width: int,
    height: int,
) -> bytes:
    grid = _load_layer_grid(settings, timestamp, layer)
    if grid is None or grid.ndim != 2:
        return _encode_png_rgba(_empty_image(width, height))

    geo = load_grid_geo(settings, timestamp=timestamp, shape=grid.shape)
    sampling_mode = "nearest" if layer in {"bathy", "unet_pred"} else "bilinear"
    sampled, inside = _sample_grid(
        grid.astype(np.float32),
        bbox,
        width,
        height,
        geo=geo,
        mode=sampling_mode,
    )

    if layer == "bathy":
        image = _render_bathy(sampled, inside)
    elif layer == "unet_pred":
        bathy_sampled = None
        bathy_grid = _load_layer_grid(settings, timestamp, "bathy")
        if bathy_grid is not None and bathy_grid.ndim == 2 and bathy_grid.shape == grid.shape:
            bathy_sampled, _ = _sample_grid(
                bathy_grid.astype(np.float32),
                bbox,
                width,
                height,
                geo=geo,
                mode="nearest",
            )
        image = _render_unet(
            np.rint(sampled).astype(np.int16),
            inside,
            bathy_blocked_sampled=bathy_sampled,
        )
    elif layer == "unet_uncertainty":
        sea_mask = None
        bathy_grid = _load_layer_grid(settings, timestamp, "bathy")
        if bathy_grid is not None and bathy_grid.ndim == 2 and bathy_grid.shape == grid.shape:
            bathy_sampled, _ = _sample_grid(
                bathy_grid.astype(np.float32),
                bbox,
                width,
                height,
                geo=geo,
                mode="nearest",
            )
            sea_mask = bathy_sampled <= 0.5
        sampled_unc = np.clip(np.nan_to_num(sampled, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        image = _render_continuous(
            sampled_unc,
            inside,
            stops=[(0.0, (16, 185, 129)), (0.5, (245, 158, 11)), (1.0, (220, 38, 38))],
            alpha_min=20,
            alpha_max=190,
            value_mask=sea_mask,
        )
    elif layer == "ais_heatmap":
        sea_mask = None
        bathy_grid = _load_layer_grid(settings, timestamp, "bathy")
        if bathy_grid is not None and bathy_grid.ndim == 2 and bathy_grid.shape == grid.shape:
            bathy_sampled, _ = _sample_grid(
                bathy_grid.astype(np.float32),
                bbox,
                width,
                height,
                geo=geo,
                mode="nearest",
            )
            sea_mask = bathy_sampled <= 0.5
        image = _render_continuous(
            sampled,
            inside,
            stops=[
                (0.0, (25, 89, 190)),
                (0.45, (56, 189, 248)),
                (0.7, (251, 191, 36)),
                (1.0, (220, 38, 38)),
            ],
            alpha_min=20,
            alpha_max=185,
            value_mask=sea_mask,
        )
    elif layer == "ice":
        sea_mask = None
        bathy_grid = _load_layer_grid(settings, timestamp, "bathy")
        if bathy_grid is not None and bathy_grid.ndim == 2 and bathy_grid.shape == grid.shape:
            bathy_sampled, _ = _sample_grid(
                bathy_grid.astype(np.float32),
                bbox,
                width,
                height,
                geo=geo,
                mode="nearest",
            )
            sea_mask = bathy_sampled <= 0.5
        sampled_ice = _normalize_ice_sampled(sampled)
        image = _render_continuous(
            sampled_ice,
            inside,
            stops=[(0.0, (147, 197, 253)), (0.5, (224, 242, 254)), (1.0, (255, 255, 255))],
            alpha_min=35,
            alpha_max=180,
            value_mask=sea_mask,
        )
    elif layer == "wave":
        sea_mask = None
        bathy_grid = _load_layer_grid(settings, timestamp, "bathy")
        if bathy_grid is not None and bathy_grid.ndim == 2 and bathy_grid.shape == grid.shape:
            bathy_sampled, _ = _sample_grid(
                bathy_grid.astype(np.float32),
                bbox,
                width,
                height,
                geo=geo,
                mode="nearest",
            )
            sea_mask = bathy_sampled <= 0.5
        image = _render_continuous(
            sampled,
            inside,
            stops=[(0.0, (20, 184, 166)), (0.5, (59, 130, 246)), (1.0, (225, 29, 72))],
            alpha_min=25,
            alpha_max=175,
            value_mask=sea_mask,
        )
    elif layer == "wind":
        sea_mask = None
        bathy_grid = _load_layer_grid(settings, timestamp, "bathy")
        if bathy_grid is not None and bathy_grid.ndim == 2 and bathy_grid.shape == grid.shape:
            bathy_sampled, _ = _sample_grid(
                bathy_grid.astype(np.float32),
                bbox,
                width,
                height,
                geo=geo,
                mode="nearest",
            )
            sea_mask = bathy_sampled <= 0.5
        image = _render_continuous(
            sampled,
            inside,
            stops=[(0.0, (16, 185, 129)), (0.5, (234, 179, 8)), (1.0, (124, 58, 237))],
            alpha_min=25,
            alpha_max=175,
            value_mask=sea_mask,
        )
    else:
        image = _empty_image(width, height)
    return _encode_png_rgba(image)


def render_tile_png(
    *,
    settings: Settings,
    timestamp: str,
    layer: str,
    z: int,
    x: int,
    y: int,
    tile_size: int = 256,
) -> bytes:
    grid = _load_layer_grid(settings, timestamp, layer)
    if grid is None or grid.ndim != 2:
        return _encode_png_rgba(_empty_image(tile_size, tile_size))

    geo = load_grid_geo(settings, timestamp=timestamp, shape=grid.shape)
    lats, lons = _tile_pixel_latlon_axes(z=z, x=x, y=y, tile_size=tile_size)
    sampling_mode = "nearest" if layer in {"bathy", "unet_pred"} else "bilinear"
    sampled, inside = _sample_grid_from_axes(
        grid.astype(np.float32),
        lats,
        lons,
        geo=geo,
        mode=sampling_mode,
    )

    if layer == "bathy":
        image = _render_bathy(sampled, inside)
    elif layer == "unet_pred":
        bathy_sampled = None
        bathy_grid = _load_layer_grid(settings, timestamp, "bathy")
        if bathy_grid is not None and bathy_grid.ndim == 2 and bathy_grid.shape == grid.shape:
            bathy_sampled, _ = _sample_grid_from_axes(
                bathy_grid.astype(np.float32),
                lats,
                lons,
                geo=geo,
                mode="nearest",
            )
        image = _render_unet(
            np.rint(sampled).astype(np.int16),
            inside,
            bathy_blocked_sampled=bathy_sampled,
        )
    elif layer == "unet_uncertainty":
        sea_mask = None
        bathy_grid = _load_layer_grid(settings, timestamp, "bathy")
        if bathy_grid is not None and bathy_grid.ndim == 2 and bathy_grid.shape == grid.shape:
            bathy_sampled, _ = _sample_grid_from_axes(
                bathy_grid.astype(np.float32),
                lats,
                lons,
                geo=geo,
                mode="nearest",
            )
            sea_mask = bathy_sampled <= 0.5
        sampled_unc = np.clip(np.nan_to_num(sampled, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        image = _render_continuous(
            sampled_unc,
            inside,
            stops=[(0.0, (16, 185, 129)), (0.5, (245, 158, 11)), (1.0, (220, 38, 38))],
            alpha_min=20,
            alpha_max=190,
            value_mask=sea_mask,
        )
    elif layer == "ais_heatmap":
        sea_mask = None
        bathy_grid = _load_layer_grid(settings, timestamp, "bathy")
        if bathy_grid is not None and bathy_grid.ndim == 2 and bathy_grid.shape == grid.shape:
            bathy_sampled, _ = _sample_grid_from_axes(
                bathy_grid.astype(np.float32),
                lats,
                lons,
                geo=geo,
                mode="nearest",
            )
            sea_mask = bathy_sampled <= 0.5
        image = _render_continuous(
            sampled,
            inside,
            stops=[
                (0.0, (25, 89, 190)),
                (0.45, (56, 189, 248)),
                (0.7, (251, 191, 36)),
                (1.0, (220, 38, 38)),
            ],
            alpha_min=20,
            alpha_max=185,
            value_mask=sea_mask,
        )
    elif layer == "ice":
        sea_mask = None
        bathy_grid = _load_layer_grid(settings, timestamp, "bathy")
        if bathy_grid is not None and bathy_grid.ndim == 2 and bathy_grid.shape == grid.shape:
            bathy_sampled, _ = _sample_grid_from_axes(
                bathy_grid.astype(np.float32),
                lats,
                lons,
                geo=geo,
                mode="nearest",
            )
            sea_mask = bathy_sampled <= 0.5
        sampled_ice = _normalize_ice_sampled(sampled)
        image = _render_continuous(
            sampled_ice,
            inside,
            stops=[(0.0, (147, 197, 253)), (0.5, (224, 242, 254)), (1.0, (255, 255, 255))],
            alpha_min=35,
            alpha_max=180,
            value_mask=sea_mask,
        )
    elif layer == "wave":
        sea_mask = None
        bathy_grid = _load_layer_grid(settings, timestamp, "bathy")
        if bathy_grid is not None and bathy_grid.ndim == 2 and bathy_grid.shape == grid.shape:
            bathy_sampled, _ = _sample_grid_from_axes(
                bathy_grid.astype(np.float32),
                lats,
                lons,
                geo=geo,
                mode="nearest",
            )
            sea_mask = bathy_sampled <= 0.5
        image = _render_continuous(
            sampled,
            inside,
            stops=[(0.0, (20, 184, 166)), (0.5, (59, 130, 246)), (1.0, (225, 29, 72))],
            alpha_min=25,
            alpha_max=175,
            value_mask=sea_mask,
        )
    elif layer == "wind":
        sea_mask = None
        bathy_grid = _load_layer_grid(settings, timestamp, "bathy")
        if bathy_grid is not None and bathy_grid.ndim == 2 and bathy_grid.shape == grid.shape:
            bathy_sampled, _ = _sample_grid_from_axes(
                bathy_grid.astype(np.float32),
                lats,
                lons,
                geo=geo,
                mode="nearest",
            )
            sea_mask = bathy_sampled <= 0.5
        image = _render_continuous(
            sampled,
            inside,
            stops=[(0.0, (16, 185, 129)), (0.5, (234, 179, 8)), (1.0, (124, 58, 237))],
            alpha_min=25,
            alpha_max=175,
            value_mask=sea_mask,
        )
    else:
        image = _empty_image(tile_size, tile_size)
    return _encode_png_rgba(image)
