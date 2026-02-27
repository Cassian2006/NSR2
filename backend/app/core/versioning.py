from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import Settings


VERSION_KEYS = ("dataset_version", "model_version", "plan_version", "eval_version")


def _safe_file_sig(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        stat = path.stat()
        return f"{int(stat.st_size)}:{int(stat.st_mtime_ns)}"
    except OSError:
        return "error"


def _annotation_pack_sig(root: Path) -> str:
    if not root.exists():
        return "missing"
    count = 0
    latest_mtime_ns = 0
    try:
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            count += 1
            try:
                mtime_ns = int(folder.stat().st_mtime_ns)
                if mtime_ns > latest_mtime_ns:
                    latest_mtime_ns = mtime_ns
            except OSError:
                continue
    except OSError:
        return "error"
    return f"{count}:{latest_mtime_ns}"


def _manifest_total_files(summary_path: Path) -> int:
    if not summary_path.exists():
        return 0
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    value = payload.get("total_files")
    try:
        return int(value)
    except Exception:
        return 0


@lru_cache(maxsize=64)
def _resolve_dataset_version_cached(
    override: str,
    manifest_summary_sig: str,
    manifest_state_sig: str,
    timestamps_sig: str,
    grid_sig: str,
    annotation_sig: str,
    total_files: int,
) -> str:
    if override.strip():
        return override.strip()
    fingerprint = "|".join(
        [
            manifest_summary_sig,
            manifest_state_sig,
            timestamps_sig,
            grid_sig,
            annotation_sig,
            str(int(total_files)),
        ]
    )
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:12]
    if total_files > 0:
        return f"ds-{total_files}-{digest}"
    return f"ds-{digest}"


def resolve_dataset_version(settings: Settings) -> str:
    manifest_summary = settings.data_root / "processed" / "manifest.summary.json"
    manifest_state = settings.data_root / "processed" / "manifest.state.json"
    return _resolve_dataset_version_cached(
        settings.dataset_version_override,
        _safe_file_sig(manifest_summary),
        _safe_file_sig(manifest_state),
        _safe_file_sig(settings.timestamps_index_path),
        _safe_file_sig(settings.grid_spec_path),
        _annotation_pack_sig(settings.annotation_pack_root),
        _manifest_total_files(manifest_summary),
    )


def build_version_snapshot(*, settings: Settings, model_version: str) -> dict[str, str]:
    return {
        "dataset_version": resolve_dataset_version(settings),
        "model_version": str(model_version).strip() or "unet_v1",
        "plan_version": str(settings.plan_version).strip() or "plan_v1",
        "eval_version": str(settings.eval_version).strip() or "eval_v1",
    }


def normalize_version_snapshot(
    payload: dict[str, Any],
    *,
    settings: Settings,
    model_version: str,
) -> dict[str, Any]:
    existing = payload.get("version_snapshot")
    existing_map = existing if isinstance(existing, dict) else {}

    snapshot = {
        "dataset_version": str(
            existing_map.get("dataset_version")
            or payload.get("dataset_version")
            or resolve_dataset_version(settings)
        ),
        "model_version": str(
            existing_map.get("model_version")
            or payload.get("model_version")
            or model_version
            or "unet_v1"
        ),
        "plan_version": str(
            existing_map.get("plan_version")
            or payload.get("plan_version")
            or settings.plan_version
            or "plan_v1"
        ),
        "eval_version": str(
            existing_map.get("eval_version")
            or payload.get("eval_version")
            or settings.eval_version
            or "eval_v1"
        ),
    }

    payload["version_snapshot"] = snapshot
    for key in VERSION_KEYS:
        payload[key] = snapshot[key]
    return payload

