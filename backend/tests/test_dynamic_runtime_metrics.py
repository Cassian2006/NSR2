from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from scripts import profile_dynamic_runtime as runtime_profile


TEST_START = {"lat": 70.5, "lon": 30.0}
TEST_GOAL = {"lat": 72.0, "lon": 150.0}


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _pick_timestamps(client: TestClient, n: int = 3) -> list[str]:
    resp = client.get("/v1/timestamps", params={"month": "all"})
    assert resp.status_code == 200
    timestamps = resp.json().get("timestamps", [])
    if len(timestamps) < 2:
        pytest.skip("Need at least two timestamps for dynamic runtime metrics test")
    return timestamps[: max(2, min(n, len(timestamps)))]


def test_dynamic_plan_exposes_runtime_monitor(client: TestClient) -> None:
    selected = _pick_timestamps(client, n=3)
    payload = {
        "timestamps": selected,
        "start": TEST_START,
        "goal": TEST_GOAL,
        "advance_steps": 8,
        "policy": {
            "objective": "shortest_distance_under_safety",
            "blocked_sources": ["bathy", "unet_blocked"],
            "caution_mode": "tie_breaker",
            "corridor_bias": 0.2,
            "smoothing": True,
            "planner": "dstar_lite",
        },
    }
    resp = client.post("/v1/route/plan/dynamic", json=payload)
    assert resp.status_code == 200
    explain = resp.json()["explain"]
    monitor = explain.get("dynamic_runtime_monitor", {})
    assert isinstance(monitor, dict)
    for key in (
        "state_load_mode",
        "state_load_workers",
        "step_wall_ms_mean",
        "step_update_ms_mean",
        "step_metrics_ms_mean",
        "step_cpu_ms_mean",
        "memory_peak_mb",
        "path_metrics_cache_hits",
        "path_metrics_cache_misses",
        "path_metrics_cache_hit_ratio",
    ):
        assert key in monitor
    assert float(monitor["step_wall_ms_mean"]) >= 0.0
    assert float(monitor["step_update_ms_mean"]) >= 0.0
    assert float(monitor["memory_peak_mb"]) >= 0.0
    assert int(monitor["path_metrics_cache_hits"]) >= 0
    assert int(monitor["path_metrics_cache_misses"]) >= 0
    assert 0.0 <= float(monitor["path_metrics_cache_hit_ratio"]) <= 1.0
    assert int(explain.get("dynamic_path_metrics_cache_hits", -1)) == int(monitor["path_metrics_cache_hits"])
    assert int(explain.get("dynamic_path_metrics_cache_misses", -1)) == int(monitor["path_metrics_cache_misses"])
    assert isinstance(explain.get("dynamic_runtime_state_load_rows"), list)


def test_runtime_profile_scoring_warn_and_fail() -> None:
    monitor_warn = {
        "step_wall_ms_mean": 1200.0,
        "step_update_ms_mean": 900.0,
        "memory_peak_mb": 512.0,
        "path_metrics_cache_hits": 1,
        "path_metrics_cache_misses": 2,
        "path_metrics_cache_hit_ratio": 0.3333,
    }
    status_warn, checks_warn = runtime_profile._score_checks(
        runtime_monitor=monitor_warn,
        max_step_wall_mean_ms=2000.0,
        max_step_update_mean_ms=1200.0,
        max_memory_peak_mb=1024.0,
        min_cache_hit_ratio=0.2,
    )
    assert status_warn == "warn"
    assert any(c.get("status") == "warn" for c in checks_warn)

    monitor_fail = {
        "step_wall_ms_mean": 2600.0,
        "step_update_ms_mean": 900.0,
        "memory_peak_mb": 512.0,
        "path_metrics_cache_hits": 10,
        "path_metrics_cache_misses": 10,
        "path_metrics_cache_hit_ratio": 0.5,
    }
    status_fail, checks_fail = runtime_profile._score_checks(
        runtime_monitor=monitor_fail,
        max_step_wall_mean_ms=2000.0,
        max_step_update_mean_ms=1200.0,
        max_memory_peak_mb=1024.0,
        min_cache_hit_ratio=0.2,
    )
    assert status_fail == "fail"
    assert any(c.get("name") == "step_wall_ms_mean" and c.get("status") == "fail" for c in checks_fail)
