from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.planning.router import _collect_path_metrics, _risk_mode_profile


class _GeoStub:
    def rc_to_latlon(self, r: int, c: int) -> tuple[float, float]:
        return float(r), float(c)


def test_risk_mode_profile_lambda_ordering() -> None:
    _, lam_cons = _risk_mode_profile("conservative")
    _, lam_bal = _risk_mode_profile("balanced")
    _, lam_agg = _risk_mode_profile("aggressive")
    assert lam_cons > lam_bal > lam_agg > 0.0


def test_collect_metrics_includes_risk_extra_cost() -> None:
    geo = _GeoStub()
    cells = [(1, 0), (1, 1), (1, 2)]
    caution = np.zeros((3, 3), dtype=bool)
    ais = np.zeros((3, 3), dtype=np.float32)
    near = np.zeros((3, 3), dtype=bool)
    risk = np.zeros((3, 3), dtype=np.float32)
    risk[1, 1] = 0.25
    risk[1, 2] = 0.45

    metrics = _collect_path_metrics(
        cells=cells,
        geo=geo,
        caution=caution,
        ais_norm=ais,
        near_blocked=near,
        caution_penalty=0.0,
        corridor_reward=0.0,
        uncertainty_penalty=None,
        risk_penalty=risk,
    )
    assert float(metrics["cost_risk_extra_km"]) > 0.0
    vals = list(metrics["risk_penalty_vals"])
    assert len(vals) >= 2
    assert max(vals) >= 0.449


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _pick_timestamp(client: TestClient) -> str:
    resp = client.get("/v1/timestamps")
    assert resp.status_code == 200
    timestamps = resp.json().get("timestamps", [])
    if not timestamps:
        pytest.skip("No timestamps available in current dataset")
    return str(timestamps[0])


def test_api_route_plan_exposes_risk_mode_fields(client: TestClient) -> None:
    ts = _pick_timestamp(client)
    base_policy = {
        "objective": "shortest_distance_under_safety",
        "blocked_sources": ["bathy", "unet_blocked"],
        "caution_mode": "tie_breaker",
        "corridor_bias": 0.2,
        "smoothing": True,
        "planner": "astar",
    }

    payload_cons = {
        "timestamp": ts,
        "start": {"lat": 70.5, "lon": 30.0},
        "goal": {"lat": 72.0, "lon": 150.0},
        "policy": {**base_policy, "risk_mode": "conservative"},
    }
    resp_cons = client.post("/v1/route/plan", json=payload_cons)
    assert resp_cons.status_code == 200
    explain_cons = resp_cons.json()["explain"]
    assert explain_cons["risk_mode"] == "conservative"
    assert "risk_lambda" in explain_cons
    assert "route_cost_risk_extra_km" in explain_cons
    assert "route_cost_effective_km" in explain_cons

    payload_agg = {
        "timestamp": ts,
        "start": {"lat": 70.5, "lon": 30.0},
        "goal": {"lat": 72.0, "lon": 150.0},
        "policy": {**base_policy, "risk_mode": "aggressive"},
    }
    resp_agg = client.post("/v1/route/plan", json=payload_agg)
    assert resp_agg.status_code == 200
    explain_agg = resp_agg.json()["explain"]
    assert explain_agg["risk_mode"] == "aggressive"
    assert float(explain_cons["risk_lambda"]) > float(explain_agg["risk_lambda"])
