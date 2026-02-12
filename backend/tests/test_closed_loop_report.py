from __future__ import annotations

import json
from pathlib import Path

from scripts.report_closed_loop_eval import aggregate_route_metrics, generate_closed_loop_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_aggregate_route_metrics_computes_safety_and_distance(tmp_path: Path) -> None:
    bench = {
        "rows": [
            {"mode": "static", "planner": "astar", "status": "ok", "distance_km": 100.0, "caution_len_km": 10.0, "runtime_ms": 50.0},
            {"mode": "static", "planner": "astar", "status": "ok", "distance_km": 120.0, "caution_len_km": 0.0, "runtime_ms": 70.0},
            {"mode": "static", "planner": "dstar_lite", "status": "ok", "distance_km": 80.0, "caution_len_km": 10.0, "runtime_ms": 30.0},
        ]
    }
    p = tmp_path / "bench.json"
    _write_json(p, bench)
    agg = aggregate_route_metrics(p, mode="static", planner="astar")
    assert agg["sample_count"] == 2
    assert abs(float(agg["distance_km"]) - 110.0) < 1e-6
    assert 0.0 <= float(agg["route_safety"]) <= 1.0


def test_generate_closed_loop_report_with_skip_inference(tmp_path: Path) -> None:
    before_train = {
        "epochs": 2,
        "best_val_loss": 0.9,
        "metrics": [{"val_miou": 0.21, "val_iou_caution": 0.11}, {"val_miou": 0.26, "val_iou_caution": 0.15}],
    }
    after_train = {
        "epochs": 2,
        "best_val_loss": 0.8,
        "metrics": [{"val_miou": 0.30, "val_iou_caution": 0.16}, {"val_miou": 0.34, "val_iou_caution": 0.22}],
    }
    before_bench = {
        "rows": [
            {"mode": "static", "planner": "astar", "status": "ok", "distance_km": 200.0, "caution_len_km": 30.0, "runtime_ms": 100.0}
        ]
    }
    after_bench = {
        "rows": [
            {"mode": "static", "planner": "astar", "status": "ok", "distance_km": 190.0, "caution_len_km": 20.0, "runtime_ms": 95.0}
        ]
    }

    p_bt = tmp_path / "before_train.json"
    p_at = tmp_path / "after_train.json"
    p_bb = tmp_path / "before_bench.json"
    p_ab = tmp_path / "after_bench.json"
    _write_json(p_bt, before_train)
    _write_json(p_at, after_train)
    _write_json(p_bb, before_bench)
    _write_json(p_ab, after_bench)

    report = generate_closed_loop_report(
        before_train_summary=p_bt,
        after_train_summary=p_at,
        before_benchmark=p_bb,
        after_benchmark=p_ab,
        mode="static",
        planner="astar",
        include_inference=False,
        infer_samples=2,
    )

    assert float(report["deltas"]["val_iou_delta"]) > 0
    assert float(report["deltas"]["route_safety_delta"]) > 0
    assert float(report["deltas"]["route_length_delta_km"]) < 0
    assert "结论" not in report["conclusion"]  # plain sentence, no heading
