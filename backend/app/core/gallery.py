from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.config import get_settings


_PLACEHOLDER_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2NcAoAAAAASUVORK5CYII="
)


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
            image_path.write_bytes(base64.b64decode(_PLACEHOLDER_PNG_BASE64))
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

