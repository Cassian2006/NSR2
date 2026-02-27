from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.planning import router as planning_router
from app.planning.router import PlanningError, _evaluate_risk_constraint, plan_grid_route


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


def _pick_plannable_timestamp() -> str:
    settings = get_settings()
    ann_root = settings.annotation_pack_root
    if not ann_root.exists():
        pytest.skip("annotation_pack root missing")
    for d in sorted(ann_root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "blocked_mask.npy").exists():
            return d.name
    pytest.skip("No timestamp with blocked_mask found in annotation_pack")


def test_risk_constraint_chance_metric_math() -> None:
    out = _evaluate_risk_constraint(
        mode="chance",
        budget=0.4,
        confidence_level=0.9,
        raw_risk_values=[0.95, 0.2, 0.91, 0.1],
    )
    assert out["metric_name"] == "chance_violation_ratio"
    assert out["sample_count"] == 4
    assert out["tail_count"] == 2
    assert pytest.approx(out["metric"], rel=1e-6) == 0.5
    assert pytest.approx(out["usage"], rel=1e-6) == 1.25
    assert out["satisfied"] is False


def test_risk_constraint_cvar_metric_math() -> None:
    out = _evaluate_risk_constraint(
        mode="cvar",
        budget=0.9,
        confidence_level=0.6,
        raw_risk_values=[0.1, 0.2, 0.8, 0.9, 1.0],
    )
    assert out["metric_name"] == "cvar_tail_mean"
    assert out["sample_count"] == 5
    assert out["tail_count"] == 2
    assert pytest.approx(out["metric"], rel=1e-6) == 0.95
    assert out["satisfied"] is False


def test_plan_grid_route_raises_when_chance_constraint_violated(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ts = _pick_plannable_timestamp()

    def _fake_risk_penalty(*, shape: tuple[int, int], risk_mode: str, **kwargs):
        penalty = np.full(shape, 0.5, dtype=np.float32)
        return penalty, {
            "risk_mode": risk_mode,
            "risk_layer": "risk_p90",
            "risk_lambda": 0.5,
            "applied": True,
        }

    monkeypatch.setattr(planning_router, "_load_risk_penalty", _fake_risk_penalty)

    with pytest.raises(PlanningError, match="Risk constraint violated"):
        plan_grid_route(
            settings=get_settings(),
            timestamp=ts,
            start=(70.5, 30.0),
            goal=(72.0, 150.0),
            model_version="unet_v1",
            corridor_bias=0.2,
            caution_mode="tie_breaker",
            smoothing=True,
            blocked_sources=["bathy", "unet_blocked"],
            planner="astar",
            uncertainty_uplift=False,
            uncertainty_uplift_scale=1.0,
            risk_mode="conservative",
            risk_weight_scale=1.0,
            risk_constraint_mode="chance",
            risk_budget=0.0,
            confidence_level=0.9,
        )


def test_api_route_plan_exposes_constraint_fields(client: TestClient) -> None:
    ts = _pick_timestamp(client)
    payload = {
        "timestamp": ts,
        "start": {"lat": 70.5, "lon": 30.0},
        "goal": {"lat": 72.0, "lon": 150.0},
        "policy": {
            "objective": "shortest_distance_under_safety",
            "blocked_sources": ["bathy", "unet_blocked"],
            "caution_mode": "tie_breaker",
            "corridor_bias": 0.2,
            "smoothing": True,
            "planner": "astar",
            "risk_mode": "balanced",
            "risk_constraint_mode": "cvar",
            "risk_budget": 1.0,
            "confidence_level": 0.9,
        },
    }
    resp = client.post("/v1/route/plan", json=payload)
    assert resp.status_code == 200
    explain = resp.json()["explain"]
    assert explain["risk_constraint_mode"] == "cvar"
    assert explain["risk_constraint_metric_name"] == "cvar_tail_mean"
    assert isinstance(explain["risk_constraint_metric"], float)
    assert isinstance(explain["risk_constraint_satisfied"], bool)
