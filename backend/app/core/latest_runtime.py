from __future__ import annotations

from threading import Lock


_LOCK = Lock()
_MAX_CONCURRENT = 2
_ACTIVE_COUNT = 0


def configure_latest_runtime(*, max_concurrent: int) -> None:
    global _MAX_CONCURRENT
    with _LOCK:
        _MAX_CONCURRENT = max(1, int(max_concurrent))


def try_acquire_slot() -> bool:
    global _ACTIVE_COUNT
    with _LOCK:
        if _ACTIVE_COUNT >= _MAX_CONCURRENT:
            return False
        _ACTIVE_COUNT += 1
        return True


def release_slot() -> None:
    global _ACTIVE_COUNT
    with _LOCK:
        if _ACTIVE_COUNT > 0:
            _ACTIVE_COUNT -= 1


def get_runtime_stats() -> dict[str, int]:
    with _LOCK:
        return {
            "max_concurrent": _MAX_CONCURRENT,
            "active": _ACTIVE_COUNT,
            "available_slots": max(0, _MAX_CONCURRENT - _ACTIVE_COUNT),
        }
