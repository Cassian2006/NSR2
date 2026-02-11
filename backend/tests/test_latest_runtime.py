from __future__ import annotations

from app.core.latest_runtime import configure_latest_runtime, get_runtime_stats, release_slot, try_acquire_slot


def test_latest_runtime_limits_concurrency() -> None:
    # Reset any leftover slots from previous tests.
    for _ in range(get_runtime_stats().get("active", 0)):
        release_slot()

    configure_latest_runtime(max_concurrent=1)
    while try_acquire_slot():
        pass

    assert try_acquire_slot() is False
    stats = get_runtime_stats()
    assert stats["max_concurrent"] == 1
    assert stats["active"] == 1
    assert stats["available_slots"] == 0

    release_slot()
    stats_after = get_runtime_stats()
    assert stats_after["active"] == 0
    assert stats_after["available_slots"] == 1

    configure_latest_runtime(max_concurrent=2)
