from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_compliance_notices_workspace_contract() -> None:
    client = TestClient(app)
    ts_resp = client.get("/v1/timestamps")
    assert ts_resp.status_code == 200
    timestamps = ts_resp.json().get("timestamps", [])
    timestamp = str(timestamps[0]) if timestamps else None

    params: dict[str, str] = {"context": "workspace"}
    if timestamp:
        params["timestamp"] = timestamp
    resp = client.get("/v1/compliance/notices", params=params)
    assert resp.status_code == 200
    payload = resp.json()

    assert payload["version"] == "compliance_v1"
    assert payload["context"] == "workspace"
    assert isinstance(payload.get("generated_at"), str)

    notices = payload.get("notices")
    assert isinstance(notices, list)
    notice_ids = {str(item.get("id")) for item in notices if isinstance(item, dict)}
    assert {"research_only", "non_navigation_instruction", "data_freshness_required"}.issubset(notice_ids)

    freshness = payload.get("data_freshness")
    assert isinstance(freshness, dict)
    assert "status" in freshness
    assert "hint" in freshness

    source_credibility = payload.get("source_credibility")
    assert isinstance(source_credibility, dict)
    assert "level" in source_credibility
    assert "summary" in source_credibility


def test_compliance_notices_context_validation() -> None:
    client = TestClient(app)
    resp = client.get("/v1/compliance/notices?context=invalid")
    assert resp.status_code == 422
