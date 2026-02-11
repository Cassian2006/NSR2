from __future__ import annotations

import time
from pathlib import Path

from app.core.config import get_settings
from app.core.latest_source_health import (
    can_attempt_source,
    configure_source_health,
    get_source_health_snapshot,
    record_source_failure,
    record_source_success,
)


def _reset_default_source_health_store() -> None:
    settings = get_settings()
    configure_source_health(
        store_path=settings.latest_source_health_path,
        failure_threshold=settings.latest_source_failure_threshold,
        cooldown_sec=settings.latest_source_cooldown_sec,
    )


def test_source_health_opens_and_recovers_circuit(tmp_path: Path) -> None:
    store = tmp_path / "source-health.json"
    try:
        configure_source_health(store_path=store, failure_threshold=2, cooldown_sec=10)
        source = "copernicus_live"

        ok, reason = can_attempt_source(source)
        assert ok is True
        assert reason == ""

        record_source_failure(source, "network_timeout")
        ok, _ = can_attempt_source(source)
        assert ok is True

        record_source_failure(source, "network_timeout_again")
        ok, reason = can_attempt_source(source)
        assert ok is False
        assert "circuit_open_retry_after_sec=" in reason

        snapshot = get_source_health_snapshot()
        state = snapshot["sources"][source]
        assert state["failure_count"] == 2
        assert state["circuit_open"] is True

        record_source_success(source)
        ok, reason = can_attempt_source(source)
        assert ok is True
        assert reason == ""
        snapshot = get_source_health_snapshot()
        assert snapshot["sources"][source]["failure_count"] == 0
    finally:
        _reset_default_source_health_store()


def test_source_health_persists_and_expires_circuit(tmp_path: Path) -> None:
    store = tmp_path / "source-health-expire.json"
    try:
        source = "remote_snapshot"
        configure_source_health(store_path=store, failure_threshold=1, cooldown_sec=1)
        record_source_failure(source, "http_500")

        # Simulate restart.
        configure_source_health(store_path=store, failure_threshold=1, cooldown_sec=1)
        ok, _ = can_attempt_source(source)
        assert ok is False

        time.sleep(1.05)
        ok, _ = can_attempt_source(source)
        assert ok is True
    finally:
        _reset_default_source_health_store()
