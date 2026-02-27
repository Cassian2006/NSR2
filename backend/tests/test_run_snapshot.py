from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.run_snapshot import (
    load_run_snapshot,
    replay_entrypoint_for_snapshot,
    save_run_snapshot,
)
from app.main import app


TEST_START = {"lat": 70.5, "lon": 30.0}
TEST_GOAL = {"lat": 72.0, "lon": 150.0}


def _settings_for_tmp(tmp_path: Path) -> Settings:
    project_root = tmp_path
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    return Settings(
        project_root=project_root,
        data_root=data_root,
        outputs_root=outputs_root,
        processed_samples_root=data_root / "processed" / "samples",
        annotation_pack_root=data_root / "processed" / "annotation_pack",
        dataset_index_path=data_root / "processed" / "dataset" / "index.json",
        timestamps_index_path=data_root / "processed" / "timestamps_index.json",
        grid_spec_path=data_root / "processed" / "grid_spec.json",
        ais_heatmap_root=data_root / "ais_heatmap",
        gallery_root=outputs_root / "gallery",
        pred_root=outputs_root / "pred",
        allow_demo_fallback=False,
        cors_origins="http://localhost:5173",
    )


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _pick_timestamp(client: TestClient) -> str:
    resp = client.get("/v1/timestamps")
    assert resp.status_code == 200
    items = resp.json().get("timestamps", [])
    if not items:
        pytest.skip("No timestamps available in current dataset")
    return str(items[0])


def test_save_run_snapshot_core_and_load(tmp_path: Path) -> None:
    settings = _settings_for_tmp(tmp_path)
    meta = save_run_snapshot(
        settings=settings,
        kind="plan",
        config={"endpoint": "/v1/route/plan", "request": {"timestamp": "2024-07-01_00"}},
        result={"distance_km": 100.0},
        version_snapshot={
            "dataset_version": "ds-test",
            "model_version": "unet_v1",
            "plan_version": "plan_v1",
            "eval_version": "eval_v1",
        },
        replay={"runner": "api.route_plan", "endpoint": "/v1/route/plan", "payload": {"timestamp": "2024-07-01_00"}},
        tags=["test"],
    )
    assert meta["snapshot_id"]
    snap = load_run_snapshot(settings=settings, snapshot_id_or_path=meta["snapshot_id"])
    assert snap["snapshot_kind"] == "plan"
    assert "runtime" in snap
    assert "dependency_versions" in snap["runtime"]
    cmd = replay_entrypoint_for_snapshot(snap, base_url="http://127.0.0.1:8000")
    assert "--snapshot-id" in cmd


def test_route_plan_returns_snapshot_and_gallery_persists(client: TestClient) -> None:
    ts = _pick_timestamp(client)
    resp = client.post(
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
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body.get("run_snapshot_id"), str)
    assert body["run_snapshot_id"] != ""
    assert isinstance(body.get("run_snapshot_file"), str)
    assert body["run_snapshot_file"].endswith(".json")

    gallery_id = body["gallery_id"]
    gallery = client.get(f"/v1/gallery/{gallery_id}")
    assert gallery.status_code == 200
    item = gallery.json()
    assert item.get("run_snapshot_id") == body["run_snapshot_id"]
    assert item.get("run_snapshot_file") == body["run_snapshot_file"]


def test_export_run_snapshot_script_basic(tmp_path: Path) -> None:
    backend_root = Path(__file__).resolve().parents[1]
    script = backend_root / "scripts" / "export_run_snapshot.py"
    env = os.environ.copy()
    env["NSR_DATA_ROOT"] = str(tmp_path / "data")
    env["NSR_OUTPUTS_ROOT"] = str(tmp_path / "outputs")
    env["NSR_ALLOW_DEMO_FALLBACK"] = "false"

    proc = subprocess.run(
        [sys.executable, str(script), "--kind", "manual_test", "--tag", "ci"],
        cwd=str(backend_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "snapshot_id=" in proc.stdout
    assert "snapshot_file=" in proc.stdout

