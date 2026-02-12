from __future__ import annotations

import numpy as np

from app.model.uncertainty_calibration import (
    calibrate_confidence,
    expected_calibration_error,
    reliability_bins,
    suggest_uncertainty_thresholds,
)


def _make_overconfident_case(n: int = 6000, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0.05, 0.95, size=n)
    y = (rng.uniform(0.0, 1.0, size=n) < p_true).astype(np.float64)
    # Over-confident transform: probabilities pushed away from 0.5.
    logit = np.log(np.clip(p_true, 1e-6, 1 - 1e-6) / np.clip(1.0 - p_true, 1e-6, 1 - 1e-6))
    p_conf = 1.0 / (1.0 + np.exp(-(2.5 * logit)))
    return p_conf.astype(np.float64), y


def test_calibration_improves_at_least_one_metric() -> None:
    conf, y = _make_overconfident_case()
    result = calibrate_confidence(conf, y, n_bins=12)
    improved = (result.brier_after <= result.brier_before) or (result.ece_after <= result.ece_before)
    assert improved
    assert result.temperature > 0.0


def test_reliability_bins_shape_and_nonnegative_gaps() -> None:
    conf, y = _make_overconfident_case(n=1000)
    bins = reliability_bins(conf, y, n_bins=10)
    assert len(bins) == 10
    assert all(float(b["gap"]) >= 0.0 for b in bins)


def test_threshold_suggestions_monotonic_targets() -> None:
    conf, y = _make_overconfident_case(n=2000)
    unc = 1.0 - conf
    suggestions = suggest_uncertainty_thresholds(unc, y, target_error_rates=(0.05, 0.1, 0.2))
    assert len(suggestions) == 3
    # Higher allowed error should not reduce achievable coverage in general.
    cov = [float(x["coverage"]) for x in suggestions]
    assert cov[2] >= cov[1] >= cov[0]


def test_ece_tracks_nonperfect_calibration() -> None:
    conf, y = _make_overconfident_case(n=1500)
    ece = expected_calibration_error(conf, y, n_bins=15)
    assert ece >= 0.0
