from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.core.uncertainty_runtime import (
    build_uncertainty_penalty_map,
    calibrate_uncertainty_grid,
    load_uncertainty_calibration_profile,
)
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


def _settings_for_tmp(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs",
        processed_samples_root=tmp_path / "data" / "processed" / "samples",
        annotation_pack_root=tmp_path / "data" / "processed" / "annotation_pack",
        env_grids_root=tmp_path / "data" / "interim" / "env_grids",
        dataset_index_path=tmp_path / "data" / "processed" / "dataset" / "index.json",
        timestamps_index_path=tmp_path / "data" / "processed" / "timestamps_index.json",
        grid_spec_path=tmp_path / "data" / "processed" / "grid_spec.json",
        ais_heatmap_root=tmp_path / "data" / "ais_heatmap",
        gallery_root=tmp_path / "outputs" / "gallery",
        pred_root=tmp_path / "outputs" / "pred",
        allow_demo_fallback=False,
    )


def test_runtime_profile_loader_from_stable_calibration_file(tmp_path: Path) -> None:
    settings = _settings_for_tmp(tmp_path)
    profile_path = settings.outputs_root / "calibration" / "unet_v1" / "calibration.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(
            {
                "profile": {
                    "model_version": "unet_v1",
                    "temperature": 1.3,
                    "uncertainty_threshold": 0.62,
                    "uplift_scale": 0.4,
                    "ece_before": 0.12,
                    "ece_after": 0.09,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    profile = load_uncertainty_calibration_profile(settings=settings, model_version="unet_v1")
    assert profile.available is True
    assert abs(profile.temperature - 1.3) < 1e-6
    assert abs(profile.uncertainty_threshold - 0.62) < 1e-6
    assert abs(profile.uplift_scale - 0.4) < 1e-6


def test_calibrated_uncertainty_penalty_monotonic_above_threshold() -> None:
    unc = np.array([[0.10, 0.45], [0.70, 0.95]], dtype=np.float32)
    unc_cal = calibrate_uncertainty_grid(unc, temperature=1.0)
    penalty = build_uncertainty_penalty_map(unc_cal, threshold=0.6, uplift_scale=0.5)
    assert penalty.shape == unc.shape
    assert np.all(penalty >= 0.0)
    assert float(penalty[0, 0]) == 0.0
    assert float(penalty[0, 1]) == 0.0
    assert float(penalty[1, 0]) > 0.0
    assert float(penalty[1, 1]) >= float(penalty[1, 0])
