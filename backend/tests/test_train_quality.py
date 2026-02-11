from __future__ import annotations

from pathlib import Path

import numpy as np

from app.model.train_quality import build_hard_sample_weights, evaluate_sample_quality


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

    weights, meta = build_hard_sample_weights([p_easy, p_hard], hard_quantile=0.5, hard_boost=3.0)
    assert weights.shape == (2,)
    assert abs(float(weights.sum()) - 1.0) < 1e-6
    assert int(meta["hard_count"]) >= 1
    assert float(weights[1]) > float(weights[0])
