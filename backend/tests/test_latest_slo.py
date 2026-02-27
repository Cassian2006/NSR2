from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_latest_sources_health_contains_slo_block() -> None:
    client = TestClient(app)
    resp = client.get("/v1/latest/sources/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert "slo" in payload
    slo = payload["slo"]
    assert str(slo.get("version")) == "latest_slo_v1"
    assert "layers" in slo
    assert "ice" in slo["layers"]


def test_latest_sources_health_accepts_timestamp() -> None:
    client = TestClient(app)
    ts_resp = client.get("/v1/timestamps")
    assert ts_resp.status_code == 200
    timestamps = ts_resp.json().get("timestamps", [])
    ts = timestamps[0] if timestamps else ""
    resp = client.get("/v1/latest/sources/health", params={"timestamp": ts})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["slo"]["timestamp"] == ts
