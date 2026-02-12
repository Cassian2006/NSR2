from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "active_learning_suggest.py"
    spec = importlib.util.spec_from_file_location("active_learning_suggest", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cap_threshold_by_ratio_caps_large_suggestions() -> None:
    mod = _load_script_module()
    caution_prob = np.array(
        [
            [0.9, 0.8, 0.7, 0.4],
            [0.85, 0.65, 0.55, 0.3],
        ],
        dtype=np.float32,
    )
    sea = np.ones_like(caution_prob, dtype=bool)

    threshold = mod._cap_threshold_by_ratio(
        caution_prob=caution_prob,
        sea=sea,
        base_threshold=0.6,
        max_ratio=0.25,
    )

    assert threshold >= 0.6
    ratio = float((caution_prob[sea] >= threshold).mean())
    assert ratio <= 0.25 + 1e-6


def test_smooth_binary_mask_removes_isolated_pixel() -> None:
    mod = _load_script_module()
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1, 1] = 1
    mask[3, 3] = 1
    mask[3, 4] = 1

    smoothed = mod._smooth_binary_mask(mask, min_neighbors=2, iters=1)

    assert smoothed[1, 1] == 0
    assert smoothed[3, 3] == 1
    assert smoothed[3, 4] == 1


def test_compute_scores_has_three_component_breakdown() -> None:
    mod = _load_script_module()
    rows = [
        {
            "timestamp": "2024-07-01_00",
            "uncertainty_raw": 0.8,
            "route_impact_raw": 0.2,
            "pred_caution_ratio": 0.03,
        },
        {
            "timestamp": "2024-07-01_06",
            "uncertainty_raw": 0.4,
            "route_impact_raw": 0.9,
            "pred_caution_ratio": 0.07,
        },
    ]
    scored = mod._compute_scores(
        rows,
        class_balance_target=0.04,
        class_balance_width=0.02,
        w_uncertainty=0.5,
        w_route_impact=0.3,
        w_class_balance=0.2,
    )
    assert len(scored) == 2
    for item in scored:
        assert "score" in item
        assert "uncertainty_score" in item
        assert "route_impact_score" in item
        assert "class_balance_score" in item
        assert 0.0 <= float(item["score"]) <= 1.0
        assert 0.0 <= float(item["uncertainty_score"]) <= 1.0
        assert 0.0 <= float(item["route_impact_score"]) <= 1.0
        assert 0.0 <= float(item["class_balance_score"]) <= 1.0


def test_score_class_balance_prefers_ratio_near_target() -> None:
    mod = _load_script_module()
    s_near = mod._score_class_balance(pred_ratio=0.04, target_ratio=0.04, width=0.02)
    s_far = mod._score_class_balance(pred_ratio=0.14, target_ratio=0.04, width=0.02)
    assert s_near > s_far
