from __future__ import annotations

import base64
import binascii
import json
import struct
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.core.config import get_settings


_PLACEHOLDER_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2NcAoAAAAASUVORK5CYII="
)
_DEFAULT_BOUNDS = {
    "min_lat": 66.0,
    "max_lat": 86.0,
    "min_lon": -30.0,
    "max_lon": 180.0,
}


def _draw_line(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    h, w = image.shape[:2]
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    if steps <= 0:
        return
    xs = np.linspace(x0, x1, steps).astype(np.int32)
    ys = np.linspace(y0, y1, steps).astype(np.int32)
    radius = max(0, thickness // 2)
    for x, y in zip(xs, ys, strict=False):
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)
        image[y_min:y_max, x_min:x_max, 0] = color[0]
        image[y_min:y_max, x_min:x_max, 1] = color[1]
        image[y_min:y_max, x_min:x_max, 2] = color[2]


def _draw_circle(
    image: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    h, w = image.shape[:2]
    x_min = max(0, cx - radius)
    x_max = min(w, cx + radius + 1)
    y_min = max(0, cy - radius)
    y_max = min(h, cy + radius + 1)
    if x_max <= x_min or y_max <= y_min:
        return
    ys, xs = np.ogrid[y_min:y_max, x_min:x_max]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2
    patch = image[y_min:y_max, x_min:x_max]
    patch[mask] = color
    image[y_min:y_max, x_min:x_max] = patch


def _extract_route_coords(payload: dict[str, Any]) -> list[tuple[float, float]]:
    route = payload.get("route_geojson")
    if not isinstance(route, dict):
        return []
    geometry = route.get("geometry")
    if not isinstance(geometry, dict):
        return []
    coords = geometry.get("coordinates")
    if not isinstance(coords, list):
        return []
    out: list[tuple[float, float]] = []
    for item in coords:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        lon = float(item[0])
        lat = float(item[1])
        out.append((lon, lat))
    return out


def _extract_point(payload: dict[str, Any], key: str) -> tuple[float, float] | None:
    data = payload.get(key)
    if not isinstance(data, dict):
        return None
    if "lat" not in data or "lon" not in data:
        return None
    return float(data["lat"]), float(data["lon"])


def _compute_bounds(
    route_coords: list[tuple[float, float]],
    start: tuple[float, float] | None,
    goal: tuple[float, float] | None,
) -> dict[str, float]:
    lats = [_DEFAULT_BOUNDS["min_lat"], _DEFAULT_BOUNDS["max_lat"]]
    lons = [_DEFAULT_BOUNDS["min_lon"], _DEFAULT_BOUNDS["max_lon"]]
    for lon, lat in route_coords:
        lats.append(lat)
        lons.append(lon)
    if start is not None:
        lats.append(start[0])
        lons.append(start[1])
    if goal is not None:
        lats.append(goal[0])
        lons.append(goal[1])
    min_lat = min(lats)
    max_lat = max(lats)
    min_lon = min(lons)
    max_lon = max(lons)
    lat_pad = max(1.0, (max_lat - min_lat) * 0.08)
    lon_pad = max(2.0, (max_lon - min_lon) * 0.08)
    return {
        "min_lat": min_lat - lat_pad,
        "max_lat": max_lat + lat_pad,
        "min_lon": min_lon - lon_pad,
        "max_lon": max_lon + lon_pad,
    }


def _to_canvas(
    bounds: dict[str, float],
    lat: float,
    lon: float,
    width: int,
    height: int,
    padding: int,
) -> tuple[int, int]:
    lon_span = max(1e-6, bounds["max_lon"] - bounds["min_lon"])
    lat_span = max(1e-6, bounds["max_lat"] - bounds["min_lat"])
    inner_w = max(1, width - padding * 2)
    inner_h = max(1, height - padding * 2)
    x = padding + ((lon - bounds["min_lon"]) / lon_span) * inner_w
    y = height - padding - ((lat - bounds["min_lat"]) / lat_span) * inner_h
    return int(round(x)), int(round(y))


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(tag)
    crc = binascii.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _encode_png_rgb(image: np.ndarray) -> bytes:
    h, w, channels = image.shape
    if channels != 3:
        raise ValueError("Expected RGB image with 3 channels")
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(h))
    png = bytearray()
    png.extend(b"\x89PNG\r\n\x1a\n")
    png.extend(_png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
    png.extend(_png_chunk(b"IDAT", zlib.compress(raw, level=6)))
    png.extend(_png_chunk(b"IEND", b""))
    return bytes(png)


def _make_placeholder_png() -> bytes:
    return base64.b64decode(_PLACEHOLDER_PNG_BASE64)


def _build_preview_png(payload: dict[str, Any]) -> bytes:
    width = 1200
    height = 700
    padding = 36
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = 232
    image[..., 1] = 240
    image[..., 2] = 248

    route_coords = _extract_route_coords(payload)
    start = _extract_point(payload, "start")
    goal = _extract_point(payload, "goal")
    bounds = _compute_bounds(route_coords, start, goal)

    # Soft ocean gradient
    gradient = np.linspace(0, 18, width, dtype=np.uint8)
    image[..., 2] = np.clip(image[..., 2] + gradient, 0, 255)

    # Grid
    for x in range(padding, width - padding + 1, 120):
        _draw_line(image, x, padding, x, height - padding, (205, 215, 225), thickness=1)
    for y in range(padding, height - padding + 1, 100):
        _draw_line(image, padding, y, width - padding, y, (205, 215, 225), thickness=1)

    # Frame
    _draw_line(image, padding, padding, width - padding, padding, (165, 178, 190), thickness=2)
    _draw_line(image, width - padding, padding, width - padding, height - padding, (165, 178, 190), thickness=2)
    _draw_line(image, width - padding, height - padding, padding, height - padding, (165, 178, 190), thickness=2)
    _draw_line(image, padding, height - padding, padding, padding, (165, 178, 190), thickness=2)

    # Route
    if len(route_coords) >= 2:
        pixels = [_to_canvas(bounds, lat, lon, width, height, padding) for lon, lat in route_coords]
        for (x0, y0), (x1, y1) in zip(pixels[:-1], pixels[1:], strict=False):
            _draw_line(image, x0, y0, x1, y1, (30, 64, 175), thickness=4)

    # Start/goal markers
    if start is not None:
        sx, sy = _to_canvas(bounds, start[0], start[1], width, height, padding)
        _draw_circle(image, sx, sy, 8, (16, 185, 129))
        _draw_circle(image, sx, sy, 3, (255, 255, 255))
    if goal is not None:
        gx, gy = _to_canvas(bounds, goal[0], goal[1], width, height, padding)
        _draw_circle(image, gx, gy, 8, (239, 68, 68))
        _draw_circle(image, gx, gy, 3, (255, 255, 255))

    return _encode_png_rgb(image)


class GalleryService:
    def __init__(self) -> None:
        settings = get_settings()
        self.run_dir = settings.gallery_root / "runs"
        self.thumb_dir = settings.gallery_root / "thumbs"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.thumb_dir.mkdir(parents=True, exist_ok=True)

    def _run_path(self, gallery_id: str) -> Path:
        return self.run_dir / f"{gallery_id}.json"

    def _image_path(self, gallery_id: str) -> Path:
        return self.thumb_dir / f"{gallery_id}.png"

    def create(self, payload: dict[str, Any]) -> str:
        gallery_id = uuid4().hex[:12]
        now = datetime.now(UTC).isoformat()
        record = {"id": gallery_id, "created_at": now, **payload}
        with self._run_path(gallery_id).open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        image_path = self._image_path(gallery_id)
        if not image_path.exists():
            try:
                image_bytes = _build_preview_png(record)
            except Exception:
                image_bytes = _make_placeholder_png()
            image_path.write_bytes(image_bytes)
        return gallery_id

    def list_items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in self.run_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            items.append(data)
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return items

    def get_item(self, gallery_id: str) -> dict[str, Any] | None:
        path = self._run_path(gallery_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def get_image_bytes(self, gallery_id: str) -> bytes | None:
        path = self._image_path(gallery_id)
        if not path.exists():
            return None
        return path.read_bytes()

    def set_image_bytes(self, gallery_id: str, image_bytes: bytes) -> bool:
        run_path = self._run_path(gallery_id)
        if not run_path.exists():
            return False
        if len(image_bytes) < 32:
            raise ValueError("image payload too small")
        if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("only PNG image is supported")
        self._image_path(gallery_id).write_bytes(image_bytes)
        return True

    def delete(self, gallery_id: str) -> bool:
        existed = False
        run_path = self._run_path(gallery_id)
        if run_path.exists():
            run_path.unlink()
            existed = True
        image_path = self._image_path(gallery_id)
        if image_path.exists():
            image_path.unlink()
            existed = True
        return existed
