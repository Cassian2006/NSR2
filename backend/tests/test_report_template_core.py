from __future__ import annotations

from app.core.report_template import build_report_template, report_template_to_csv, report_template_to_markdown


def test_report_template_core_sections_and_serialization() -> None:
    gallery_item = {
        "id": "abc123",
        "timestamp": "2024-07-01_00",
        "start": {"lat": 70.5, "lon": 30.0},
        "goal": {"lat": 72.0, "lon": 150.0},
        "distance_km": 1000.0,
        "caution_len_km": 100.0,
        "model_version": "unet_v1",
        "action": {
            "policy": {
                "objective": "shortest_distance_under_safety",
                "blocked_sources": ["bathy", "unet_blocked"],
                "caution_mode": "tie_breaker",
                "smoothing": True,
                "corridor_bias": 0.2,
                "risk_mode": "balanced",
                "risk_constraint_mode": "none",
            }
        },
        "explain": {
            "distance_km": 1000.0,
            "distance_nm": 540.0,
            "caution_len_km": 100.0,
            "corridor_alignment": 0.45,
            "planner": "astar",
            "route_points": 120,
        },
        "route_geojson": {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[30.0, 70.5], [150.0, 72.0]]}},
    }
    risk_report = {
        "risk": {"risk_exposure": 3.1, "high_risk_crossing_ratio": 0.12},
        "candidate_comparison": {"count": 3, "ok_count": 3, "pareto_summary": {"frontier_count": 2}},
    }
    compliance = {"version": "compliance_v1", "data_freshness": {"status": "fresh"}, "source_credibility": {"level": "normal"}}

    report = build_report_template(gallery_item=gallery_item, risk_report=risk_report, compliance=compliance)
    assert report["template_version"] == "report_template_v1"
    for key in ["method", "data", "results", "statistics", "limitations", "reproducibility", "compliance"]:
        assert key in report

    csv_payload = report_template_to_csv(report)
    assert "section,field,value" in csv_payload
    assert "method" in csv_payload

    md_payload = report_template_to_markdown(report)
    assert "## 方法" in md_payload
    assert "## 复现" in md_payload
