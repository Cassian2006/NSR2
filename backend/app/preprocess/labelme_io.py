from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _shape_points(shape: dict[str, Any]) -> list[tuple[float, float]]:
    pts = shape.get("points", [])
    out: list[tuple[float, float]] = []
    for p in pts:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append((float(p[0]), float(p[1])))
    return out


def labelme_json_to_binary_mask(
    json_path: Path,
    target_label: str = "caution",
) -> np.ndarray:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    h = int(data["imageHeight"])
    w = int(data["imageWidth"])
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)

    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        if label != target_label:
            continue
        shape_type = str(shape.get("shape_type", "polygon")).strip().lower()
        pts = _shape_points(shape)
        if not pts:
            continue

        if shape_type == "rectangle" and len(pts) >= 2:
            draw.rectangle([pts[0], pts[1]], fill=1)
        elif shape_type == "circle" and len(pts) >= 2:
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            r = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), fill=1)
        else:
            draw.polygon(pts, fill=1)

    return np.array(canvas, dtype=np.uint8)


def save_binary_mask(mask: np.ndarray, png_path: Path, npy_path: Path) -> None:
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(png_path)
    np.save(npy_path, (mask > 0).astype(np.uint8))
