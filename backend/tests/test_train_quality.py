from __future__ import annotations

from pathlib import Path

import numpy as np

from app.model.train_quality import build_hard_sample_weights, evaluate_sample_quality, sample_indices_from_weights
from scripts.qc_unet_manifest import evaluate_annotation_quality


def test_evaluate_sample_quality_detects_invalid_sample() -> None:
    x = np.random.default_rng(0).normal(size=(7, 8, 8)).astype(np.float32)
    y = np.zeros((8, 8), dtype=np.int64)
    y[0, 0] = 9
    qc = evaluate_sample_quality(x, y, min_foreground_ratio=0.01, max_nan_ratio=0.0)
    assert qc.ok is False
    assert any("invalid_label_values" in r for r in qc.reasons)


def test_build_hard_sample_weights_boosts_harder_masks(tmp_path: Path) -> None:
    y_easy = np.zeros((32, 32), dtype=np.uint8)
    y_hard = np.zeros((32, 32), dtype=np.uint8)
    y_hard[::2, ::2] = 1
    y_hard[1::2, 1::2] = 2

    p_easy = tmp_path / "easy.npy"
    p_hard = tmp_path / "hard.npy"
    np.save(p_easy, y_easy)
    np.save(p_hard, y_hard)

    weights, meta, hard_mask = build_hard_sample_weights([p_easy, p_hard], hard_quantile=0.5, hard_boost=3.0)
    assert weights.shape == (2,)
    assert abs(float(weights.sum()) - 1.0) < 1e-6
    assert int(meta["hard_count"]) >= 1
    assert float(weights[1]) > float(weights[0])
    assert hard_mask.shape == (2,)
    assert bool(hard_mask[1])


def test_hard_sample_weight_ratio_controls_sampling_distribution(tmp_path: Path) -> None:
    paths: list[Path] = []
    for i in range(8):
        y = np.zeros((24, 24), dtype=np.uint8)
        if i >= 5:
            y[::2, ::2] = 1
            y[1::2, 1::2] = 2
        p = tmp_path / f"s{i}.npy"
        np.save(p, y)
        paths.append(p)

    weights, meta, hard_mask = build_hard_sample_weights(
        paths,
        hard_quantile=0.6,
        hard_boost=4.0,
        hard_target_ratio=0.55,
        hard_max_ratio=0.60,
    )
    draws = sample_indices_from_weights(weights, 6000, seed=123)
    observed = float(hard_mask[draws].mean())

    assert observed <= 0.64
    assert abs(observed - float(meta["hard_weight_ratio"])) < 0.05
    assert float(meta["hard_weight_ratio"]) <= float(meta["hard_max_ratio"]) + 1e-6


def test_evaluate_annotation_quality_detects_empty_and_conflict() -> None:
    y = np.zeros((16, 16), dtype=np.uint8)
    blocked = np.zeros((16, 16), dtype=np.uint8)
    blocked[4:12, 4:12] = 1
    caution = np.zeros((16, 16), dtype=np.uint8)

    qc = evaluate_annotation_quality(
        y_class=y,
        blocked_mask=blocked,
        caution_mask=caution,
        min_component_pixels=8,
        max_tiny_components=3,
        max_tiny_components_ratio=0.5,
        max_conflict_ratio=0.0,
    )
    assert "empty_caution_annotation" in qc.reasons

    caution2 = np.zeros((16, 16), dtype=np.uint8)
    caution2[6:10, 6:10] = 1
    y2 = np.where(blocked > 0, 2, 0).astype(np.uint8)
    y2[caution2 > 0] = 1
    qc2 = evaluate_annotation_quality(
        y_class=y2,
        blocked_mask=blocked,
        caution_mask=caution2,
        min_component_pixels=2,
        max_tiny_components=100,
        max_tiny_components_ratio=1.0,
        max_conflict_ratio=0.0,
    )
    assert any(r.startswith("caution_blocked_conflict_ratio>") for r in qc2.reasons)


def test_evaluate_annotation_quality_detects_tiny_caution_regions() -> None:
    y = np.zeros((20, 20), dtype=np.uint8)
    blocked = np.zeros((20, 20), dtype=np.uint8)
    caution = np.zeros((20, 20), dtype=np.uint8)
    caution[1, 1] = 1
    caution[3, 3] = 1
    caution[5, 5] = 1
    caution[7, 7] = 1
    y[caution > 0] = 1

    qc = evaluate_annotation_quality(
        y_class=y,
        blocked_mask=blocked,
        caution_mask=caution,
        min_component_pixels=4,
        max_tiny_components=2,
        max_tiny_components_ratio=0.2,
        max_conflict_ratio=0.0,
    )
    assert any(r.startswith("caution_tiny_regions>") for r in qc.reasons)
