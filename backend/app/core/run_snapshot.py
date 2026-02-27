from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from app.core.config import Settings


_DEP_KEYS = (
    "fastapi",
    "numpy",
    "pydantic",
    "pydantic-settings",
    "uvicorn",
    "torch",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:10]


def _safe_git_commit(project_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _dependency_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for key in _DEP_KEYS:
        try:
            out[key] = metadata.version(key)
        except metadata.PackageNotFoundError:
            out[key] = "not_installed"
        except Exception:
            out[key] = "unknown"
    return out


def _snapshot_root(settings: Settings) -> Path:
    root = settings.outputs_root / "repro" / "run_snapshots"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_snapshot_id(kind: str, *, hint: dict[str, Any] | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = _short_hash(hint or {"kind": kind, "stamp": stamp})
    return f"{kind}_{stamp}_{suffix}"


def save_run_snapshot(
    *,
    settings: Settings,
    kind: str,
    config: dict[str, Any],
    result: dict[str, Any],
    version_snapshot: dict[str, Any] | None = None,
    replay: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    snapshot_id = build_snapshot_id(kind, hint={"kind": kind, "config": config})
    root = _snapshot_root(settings)
    path = root / f"{snapshot_id}.json"
    payload = {
        "snapshot_id": snapshot_id,
        "snapshot_kind": str(kind),
        "created_at": _utc_now_iso(),
        "project_root": str(settings.project_root),
        "git_commit": _safe_git_commit(settings.project_root),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "dependency_versions": _dependency_versions(),
        },
        "version_snapshot": dict(version_snapshot or {}),
        "config": dict(config),
        "result": dict(result),
        "replay": dict(replay or {}),
        "tags": list(tags or []),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_ptr = root / f"latest_{kind}.json"
    latest_ptr.write_text(
        json.dumps(
            {
                "snapshot_id": snapshot_id,
                "snapshot_kind": str(kind),
                "path": str(path),
                "created_at": payload["created_at"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"snapshot_id": snapshot_id, "snapshot_file": str(path)}


def resolve_snapshot_path(settings: Settings, snapshot_id_or_path: str) -> Path:
    candidate = Path(snapshot_id_or_path)
    if candidate.exists():
        return candidate
    root = _snapshot_root(settings)
    by_id = root / f"{snapshot_id_or_path}.json"
    if by_id.exists():
        return by_id
    raise FileNotFoundError(f"snapshot not found: {snapshot_id_or_path}")


def load_run_snapshot(*, settings: Settings, snapshot_id_or_path: str) -> dict[str, Any]:
    path = resolve_snapshot_path(settings, snapshot_id_or_path)
    return json.loads(path.read_text(encoding="utf-8"))


def replay_entrypoint_for_snapshot(snapshot: dict[str, Any], *, base_url: str = "http://127.0.0.1:8000") -> str:
    snapshot_id = str(snapshot.get("snapshot_id", ""))
    return f"python scripts/export_run_snapshot.py --snapshot-id {snapshot_id} --replay --base-url {base_url}"

