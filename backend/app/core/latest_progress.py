from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any


_LOCK = Lock()
_STATE: dict[str, dict[str, Any]] = {}
_STORE_PATH: Path | None = None
_RETENTION_HOURS = 72
_MAX_ENTRIES = 2000


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _new_state(progress_id: str, message: str) -> dict[str, Any]:
    now = _now_iso()
    return {
        "progress_id": progress_id,
        "status": "running",
        "phase": "init",
        "message": message,
        "percent": 1,
        "error": None,
        "created_at": now,
        "updated_at": now,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _load_locked() -> None:
    if _STORE_PATH is None or not _STORE_PATH.exists():
        return
    try:
        raw = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(raw, dict):
        return
    items = raw.get("items")
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("progress_id", "")).strip()
        if not pid:
            continue
        _STATE[pid] = item


def _mark_recovered_running_locked() -> None:
    now = _now_iso()
    for payload in _STATE.values():
        if payload.get("status") != "running":
            continue
        payload["status"] = "failed"
        payload["phase"] = "recovered"
        payload["message"] = "Recovered after restart: previous latest job was interrupted."
        payload["error"] = "interrupted_by_restart"
        payload["updated_at"] = now
        payload["percent"] = max(1, int(payload.get("percent", 1)))


def _prune_locked() -> None:
    if not _STATE:
        return
    cutoff = _now_utc() - timedelta(hours=max(1, int(_RETENTION_HOURS)))
    for pid in list(_STATE.keys()):
        updated_at = _parse_iso(str(_STATE[pid].get("updated_at", "")))
        if updated_at is not None and updated_at < cutoff:
            _STATE.pop(pid, None)

    if len(_STATE) <= _MAX_ENTRIES:
        return
    ranked = sorted(
        _STATE.items(),
        key=lambda kv: (_parse_iso(str(kv[1].get("updated_at", ""))) or datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True,
    )
    keep_ids = {pid for pid, _ in ranked[:_MAX_ENTRIES]}
    for pid in list(_STATE.keys()):
        if pid not in keep_ids:
            _STATE.pop(pid, None)


def _flush_locked() -> None:
    if _STORE_PATH is None:
        return
    _write_json_atomic(
        _STORE_PATH,
        {
            "updated_at": _now_iso(),
            "retention_hours": _RETENTION_HOURS,
            "max_entries": _MAX_ENTRIES,
            "items": list(_STATE.values()),
        },
    )


def configure_progress_store(
    *,
    store_path: Path,
    retention_hours: int,
    max_entries: int,
) -> None:
    global _STORE_PATH, _RETENTION_HOURS, _MAX_ENTRIES
    with _LOCK:
        _STORE_PATH = store_path
        _RETENTION_HOURS = max(1, int(retention_hours))
        _MAX_ENTRIES = max(100, int(max_entries))
        _STATE.clear()
        _load_locked()
        _mark_recovered_running_locked()
        _prune_locked()
        _flush_locked()


def start_progress(progress_id: str, *, message: str = "Starting latest data flow") -> dict[str, Any]:
    payload = _new_state(progress_id, message)
    with _LOCK:
        _STATE[progress_id] = payload
        _prune_locked()
        _flush_locked()
    return payload


def update_progress(
    progress_id: str,
    *,
    phase: str | None = None,
    message: str | None = None,
    percent: int | None = None,
) -> dict[str, Any]:
    with _LOCK:
        cur = _STATE.get(progress_id)
        if cur is None:
            cur = _new_state(progress_id, "Starting latest data flow")
            _STATE[progress_id] = cur

        if phase is not None:
            cur["phase"] = phase
        if message is not None:
            cur["message"] = message
        if percent is not None:
            cur["percent"] = max(0, min(100, int(percent)))
        cur["updated_at"] = _now_iso()
        _prune_locked()
        _flush_locked()
        return dict(cur)


def complete_progress(progress_id: str, *, message: str = "Completed") -> dict[str, Any]:
    with _LOCK:
        cur = _STATE.get(progress_id)
        if cur is None:
            cur = _new_state(progress_id, "Starting latest data flow")
            _STATE[progress_id] = cur
        cur["status"] = "completed"
        cur["phase"] = "done"
        cur["message"] = message
        cur["percent"] = 100
        cur["error"] = None
        cur["updated_at"] = _now_iso()
        _prune_locked()
        _flush_locked()
        return dict(cur)


def fail_progress(progress_id: str, *, error: str, phase: str = "error") -> dict[str, Any]:
    with _LOCK:
        cur = _STATE.get(progress_id)
        if cur is None:
            cur = _new_state(progress_id, "Starting latest data flow")
            _STATE[progress_id] = cur
        cur["status"] = "failed"
        cur["phase"] = phase
        cur["message"] = "Latest data flow failed"
        cur["error"] = error
        cur["updated_at"] = _now_iso()
        cur["percent"] = max(1, int(cur.get("percent", 1)))
        _prune_locked()
        _flush_locked()
        return dict(cur)


def get_progress(progress_id: str) -> dict[str, Any]:
    with _LOCK:
        cur = _STATE.get(progress_id)
        if cur is None:
            return {
                "progress_id": progress_id,
                "exists": False,
                "status": "not_found",
                "phase": "unknown",
                "message": "Progress not found",
                "percent": 0,
                "error": None,
                "updated_at": _now_iso(),
            }
        payload = dict(cur)
        payload["exists"] = True
        return payload
