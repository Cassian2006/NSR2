from __future__ import annotations

import json
from pathlib import Path

from app.core.config import get_settings
from app.core.latest_progress import (
    complete_progress,
    configure_progress_store,
    get_progress,
    start_progress,
    update_progress,
)


def _reset_default_progress_store() -> None:
    settings = get_settings()
    configure_progress_store(
        store_path=settings.latest_progress_store_path,
        retention_hours=settings.latest_progress_retention_hours,
        max_entries=settings.latest_progress_max_entries,
    )


def test_progress_store_persists_and_recovers_running_state(tmp_path: Path) -> None:
    store = tmp_path / "progress.json"
    try:
        configure_progress_store(store_path=store, retention_hours=24, max_entries=200)

        start_progress("p-recover", message="starting")
        update_progress("p-recover", phase="download", message="in-flight", percent=35)
        assert store.exists()

        # Simulate process restart by re-configuring from persisted store.
        configure_progress_store(store_path=store, retention_hours=24, max_entries=200)
        payload = get_progress("p-recover")
        assert payload["exists"] is True
        assert payload["status"] == "failed"
        assert payload["phase"] == "recovered"
        assert payload["error"] == "interrupted_by_restart"
    finally:
        _reset_default_progress_store()


def test_progress_store_prunes_by_max_entries(tmp_path: Path) -> None:
    store = tmp_path / "progress-prune.json"
    try:
        configure_progress_store(store_path=store, retention_hours=24, max_entries=100)

        for idx in range(130):
            pid = f"p-{idx:03d}"
            start_progress(pid, message="start")
            complete_progress(pid, message="done")

        payload = json.loads(store.read_text(encoding="utf-8"))
        assert isinstance(payload.get("items"), list)
        assert len(payload["items"]) <= 100
    finally:
        _reset_default_progress_store()
