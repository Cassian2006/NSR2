from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


def _prepare_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "candidate_count": 12,
                "top_k": 2,
                "labelme_dir": str(run_dir / "labelme_active_topk"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    labelme = run_dir / "labelme_active_topk"
    labelme.mkdir(parents=True, exist_ok=True)
    with (labelme / "mapping.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "filename",
                "timestamp",
                "score",
                "uncertainty_score",
                "route_impact_score",
                "class_balance_score",
                "pred_caution_ratio",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "rank": 1,
                "filename": "active_001.png",
                "timestamp": "2024-07-01_00",
                "score": 0.91,
                "uncertainty_score": 0.9,
                "route_impact_score": 0.8,
                "class_balance_score": 0.7,
                "pred_caution_ratio": 0.04,
            }
        )
    explain_dir = run_dir / "explanations"
    explain_dir.mkdir(parents=True, exist_ok=True)
    (explain_dir / "active_001_explain.json").write_text(
        json.dumps(
            {
                "dominant_factor": "ice_contribution",
                "factors_norm": {
                    "ice_contribution": 0.4,
                    "wave_contribution": 0.2,
                    "wind_contribution": 0.1,
                    "ais_deviation": 0.2,
                    "historical_misclassification_risk": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )


def test_active_review_endpoints_roundtrip() -> None:
    settings = get_settings()
    run_root = settings.outputs_root / "active_learning"
    test_run = run_root / "active_test_api"
    state_file = run_root / "review_state" / "active_test_api.json"
    if test_run.exists():
        shutil.rmtree(test_run)
    if state_file.exists():
        state_file.unlink()
    _prepare_run(test_run)

    try:
        client = TestClient(app)
        runs_resp = client.get("/v1/active/review/runs")
        assert runs_resp.status_code == 200
        runs = runs_resp.json().get("runs", [])
        assert any(r.get("run_id") == "active_test_api" for r in runs)

        items_resp = client.get("/v1/active/review/items", params={"run_id": "active_test_api", "limit": 5})
        assert items_resp.status_code == 200
        payload = items_resp.json()
        assert payload["run_id"] == "active_test_api"
        assert payload["items"]
        first = payload["items"][0]
        assert first["timestamp"] == "2024-07-01_00"
        assert "explanation" in first

        decision_resp = client.post(
            "/v1/active/review/decision",
            json={
                "run_id": "active_test_api",
                "timestamp": "2024-07-01_00",
                "decision": "accepted",
                "note": "looks good",
            },
        )
        assert decision_resp.status_code == 200
        assert decision_resp.json()["ok"] is True
        assert state_file.exists()

        items_resp2 = client.get("/v1/active/review/items", params={"run_id": "active_test_api", "limit": 5})
        assert items_resp2.status_code == 200
        first2 = items_resp2.json()["items"][0]
        assert first2["decision"]["decision"] == "accepted"
    finally:
        if test_run.exists():
            shutil.rmtree(test_run)
        if state_file.exists():
            state_file.unlink()

