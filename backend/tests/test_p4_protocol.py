from __future__ import annotations

from app.core.config import get_settings
from app.core.p4_protocol import build_p4_eval_protocol, compare_repeat_rows, compute_protocol_hash


def test_protocol_hash_stable_for_same_payload() -> None:
    payload = {"a": 1, "b": {"x": [1, 2, 3]}}
    h1 = compute_protocol_hash(payload)
    h2 = compute_protocol_hash(payload)
    assert h1 == h2
    assert len(h1) == 16


def test_compare_repeat_rows_detects_violation() -> None:
    rows_a = [{"case_id": "S001/astar", "distance_km": 100.0, "route_cost_effective_km": 105.0, "caution_len_km": 2.0}]
    rows_b = [{"case_id": "S001/astar", "distance_km": 110.0, "route_cost_effective_km": 106.0, "caution_len_km": 2.0}]
    report = compare_repeat_rows(
        rows_a,
        rows_b,
        metrics=["distance_km", "route_cost_effective_km", "caution_len_km"],
        abs_tol=1e-6,
        rel_tol=1e-6,
    )
    assert report["status"] == "fail"
    assert report["case_count"] == 1


def test_build_p4_eval_protocol_contains_required_fields() -> None:
    settings = get_settings()
    payload = build_p4_eval_protocol(
        settings=settings,
        model_version="unet_v1",
        static_case_count=3,
        dynamic_case_count=2,
        dynamic_window=3,
        dynamic_advance_steps=8,
    )
    assert payload["protocol_version"] == "p4_eval_protocol_v1"
    assert isinstance(payload.get("protocol_hash"), str) and len(payload["protocol_hash"]) == 16
    assert "version_snapshot" in payload
    assert "scenarios" in payload
    assert isinstance(payload["scenarios"].get("static", []), list)
    assert isinstance(payload["scenarios"].get("dynamic", []), list)
    if payload["scope"]["timestamp_count"] >= 3:
        assert len(payload["scenarios"]["static"]) >= 1
