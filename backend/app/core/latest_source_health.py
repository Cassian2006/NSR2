from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock


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


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{_now_utc().timestamp():.6f}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


@dataclass
class _SourceState:
    failure_count: int = 0
    last_error: str = ""
    last_failure_at: str | None = None
    last_success_at: str | None = None
    circuit_open_until: str | None = None

    def as_dict(self) -> dict:
        return {
            "failure_count": int(self.failure_count),
            "last_error": self.last_error,
            "last_failure_at": self.last_failure_at,
            "last_success_at": self.last_success_at,
            "circuit_open_until": self.circuit_open_until,
        }


_LOCK = Lock()
_STORE_PATH: Path | None = None
_FAILURE_THRESHOLD: int = 3
_COOLDOWN_SEC: int = 900
_SOURCES: dict[str, _SourceState] = {}


def _coerce_state(raw: dict) -> _SourceState:
    return _SourceState(
        failure_count=max(0, int(raw.get("failure_count", 0))),
        last_error=str(raw.get("last_error", "")),
        last_failure_at=raw.get("last_failure_at"),
        last_success_at=raw.get("last_success_at"),
        circuit_open_until=raw.get("circuit_open_until"),
    )


def _source_key(source: str) -> str:
    return source.strip().lower()


def _persist_unlocked() -> None:
    if _STORE_PATH is None:
        return
    payload = {
        "updated_at": _now_iso(),
        "failure_threshold": int(_FAILURE_THRESHOLD),
        "cooldown_sec": int(_COOLDOWN_SEC),
        "sources": {name: state.as_dict() for name, state in sorted(_SOURCES.items())},
    }
    _atomic_write_json(_STORE_PATH, payload)


def _load_unlocked() -> None:
    _SOURCES.clear()
    if _STORE_PATH is None or not _STORE_PATH.exists():
        return
    try:
        payload = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, dict):
        return
    for source, raw_state in raw_sources.items():
        if isinstance(raw_state, dict):
            _SOURCES[_source_key(str(source))] = _coerce_state(raw_state)


def configure_source_health(*, store_path: Path, failure_threshold: int, cooldown_sec: int) -> None:
    global _STORE_PATH, _FAILURE_THRESHOLD, _COOLDOWN_SEC
    with _LOCK:
        _STORE_PATH = store_path
        _FAILURE_THRESHOLD = max(1, int(failure_threshold))
        _COOLDOWN_SEC = max(1, int(cooldown_sec))
        _load_unlocked()
        _persist_unlocked()


def _get_state_unlocked(source: str) -> _SourceState:
    key = _source_key(source)
    state = _SOURCES.get(key)
    if state is None:
        state = _SourceState()
        _SOURCES[key] = state
    return state


def can_attempt_source(source: str) -> tuple[bool, str]:
    with _LOCK:
        state = _get_state_unlocked(source)
        open_until = _parse_iso(state.circuit_open_until)
        if open_until is None:
            return True, ""
        now = _now_utc()
        if open_until <= now:
            state.circuit_open_until = None
            _persist_unlocked()
            return True, ""
        retry_after = int((open_until - now).total_seconds())
        return False, f"circuit_open_retry_after_sec={max(1, retry_after)}"


def record_source_success(source: str) -> None:
    with _LOCK:
        state = _get_state_unlocked(source)
        state.failure_count = 0
        state.last_error = ""
        state.last_success_at = _now_iso()
        state.circuit_open_until = None
        _persist_unlocked()


def record_source_failure(source: str, error: str) -> None:
    with _LOCK:
        state = _get_state_unlocked(source)
        now = _now_utc()
        state.failure_count += 1
        state.last_error = str(error)
        state.last_failure_at = now.isoformat()
        if state.failure_count >= _FAILURE_THRESHOLD:
            state.circuit_open_until = (now + timedelta(seconds=_COOLDOWN_SEC)).isoformat()
        _persist_unlocked()


def get_source_health_snapshot() -> dict:
    with _LOCK:
        now = _now_utc()
        sources: dict[str, dict] = {}
        for source, state in sorted(_SOURCES.items()):
            open_until = _parse_iso(state.circuit_open_until)
            retry_after = 0
            is_open = False
            if open_until is not None and open_until > now:
                is_open = True
                retry_after = max(1, int((open_until - now).total_seconds()))
            sources[source] = {
                **state.as_dict(),
                "circuit_open": is_open,
                "retry_after_sec": retry_after,
            }
        return {
            "updated_at": _now_iso(),
            "failure_threshold": int(_FAILURE_THRESHOLD),
            "cooldown_sec": int(_COOLDOWN_SEC),
            "sources": sources,
        }
