from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.p4_repro_audit import (
    audit_snapshot_replay,
    compare_metric_drift,
    summarize_repro_audit,
    validate_snapshot_binding,
)
from app.core.run_snapshot import load_run_snapshot
from app.main import app


TEST_START = {"lat": 70.5, "lon": 30.0}
TEST_GOAL = {"lat": 72.0, "lon": 150.0}


def test_compare_metric_drift_status_levels() -> None:
    pass_case = compare_metric_drift(baseline=100.0, replayed=100.4, abs_tol=0.5, rel_tol=0.01)
    warn_case = compare_metric_drift(baseline=100.0, replayed=100.9, abs_tol=0.5, rel_tol=0.01)
    fail_case = compare_metric_drift(baseline=100.0, replayed=102.5, abs_tol=0.5, rel_tol=0.01)

    assert pass_case["status"] == "PASS"
    assert warn_case["status"] == "WARN"
    assert fail_case["status"] == "FAIL"


def test_validate_snapshot_binding_missing_replay() -> None:
    res = validate_snapshot_binding({"snapshot_id": "x"})
    assert res["ok"] is False
    assert res["reason"] == "missing_replay_block"


def test_audit_snapshot_replay_for_route_plan() -> None:
    with TestClient(app) as client:
        ts_resp = client.get("/v1/timestamps")
        assert ts_resp.status_code == 200
        timestamps = ts_resp.json().get("timestamps", [])
        if not timestamps:
            pytest.skip("No timestamps available in current dataset")
        ts = str(timestamps[0])

        plan_resp = client.post(
            "/v1/route/plan",
            json={
                "timestamp": ts,
                "start": TEST_START,
                "goal": TEST_GOAL,
                "policy": {
                    "objective": "shortest_distance_under_safety",
                    "blocked_sources": ["bathy", "unet_blocked"],
                    "caution_mode": "tie_breaker",
                    "corridor_bias": 0.2,
                    "smoothing": True,
                    "planner": "astar",
                },
            },
        )
        assert plan_resp.status_code == 200
        snapshot_id = str(plan_resp.json().get("run_snapshot_id", ""))
        assert snapshot_id

        snapshot = load_run_snapshot(settings=get_settings(), snapshot_id_or_path=snapshot_id)
        audit = audit_snapshot_replay(snapshot=snapshot, client=client)

    assert audit["status"] in {"PASS", "WARN"}
    assert audit["replay"]["http_status"] == 200
    assert "metrics" in audit


def test_summarize_repro_audit() -> None:
    summary = summarize_repro_audit(
        [
            {"status": "PASS"},
            {"status": "WARN"},
            {"status": "FAIL"},
        ]
    )
    assert summary["overall_status"] == "FAIL"
    assert summary["count"] == 3
