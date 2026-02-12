from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clip01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), eps, 1.0 - eps)


def brier_score(confidence: np.ndarray, outcome: np.ndarray) -> float:
    p = _clip01(confidence)
    y = np.asarray(outcome, dtype=np.float64)
    if p.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(confidence: np.ndarray, outcome: np.ndarray, n_bins: int = 15) -> float:
    p = _clip01(confidence)
    y = np.asarray(outcome, dtype=np.float64)
    if p.size == 0:
        return 0.0
    n_bins = max(4, int(n_bins))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        w = float(mask.mean())
        ece += w * abs(acc - conf)
    return float(ece)


def apply_temperature_scaling(confidence: np.ndarray, temperature: float) -> np.ndarray:
    t = max(1e-4, float(temperature))
    p = _clip01(confidence)
    logits = np.log(p / (1.0 - p))
    scaled = 1.0 / (1.0 + np.exp(-(logits / t)))
    return _clip01(scaled)


def _binary_nll(confidence: np.ndarray, outcome: np.ndarray) -> float:
    p = _clip01(confidence)
    y = np.asarray(outcome, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_temperature(confidence: np.ndarray, outcome: np.ndarray) -> float:
    p = _clip01(confidence)
    y = np.asarray(outcome, dtype=np.float64)
    if p.size == 0:
        return 1.0
    # Coarse-to-fine grid search; stable and dependency-free.
    grid = np.exp(np.linspace(np.log(0.2), np.log(8.0), 121))
    best_t = 1.0
    best_loss = float("inf")
    for t in grid:
        cal = apply_temperature_scaling(p, float(t))
        nll = _binary_nll(cal, y)
        if nll < best_loss:
            best_loss = nll
            best_t = float(t)
    return best_t


@dataclass(frozen=True)
class CalibrationResult:
    temperature: float
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float
    nll_before: float
    nll_after: float
    improved_metric: str
    confidence_after: np.ndarray


def calibrate_confidence(confidence: np.ndarray, outcome: np.ndarray, n_bins: int = 15) -> CalibrationResult:
    p = _clip01(confidence)
    y = np.asarray(outcome, dtype=np.float64)
    t = fit_temperature(p, y)
    cal = apply_temperature_scaling(p, t)
    ece_before = expected_calibration_error(p, y, n_bins=n_bins)
    ece_after = expected_calibration_error(cal, y, n_bins=n_bins)
    brier_before = brier_score(p, y)
    brier_after = brier_score(cal, y)
    nll_before = _binary_nll(p, y)
    nll_after = _binary_nll(cal, y)

    # Guardrail: if fitted temperature worsens both Brier and ECE, keep identity.
    if (brier_after > brier_before) and (ece_after > ece_before):
        t = 1.0
        cal = p.copy()
        ece_after = ece_before
        brier_after = brier_before
        nll_after = nll_before

    if brier_after <= brier_before:
        improved = "brier"
    elif ece_after <= ece_before:
        improved = "ece"
    else:
        improved = "none"

    return CalibrationResult(
        temperature=t,
        ece_before=ece_before,
        ece_after=ece_after,
        brier_before=brier_before,
        brier_after=brier_after,
        nll_before=nll_before,
        nll_after=nll_after,
        improved_metric=improved,
        confidence_after=cal,
    )


def reliability_bins(confidence: np.ndarray, outcome: np.ndarray, n_bins: int = 15) -> list[dict[str, float]]:
    p = _clip01(confidence)
    y = np.asarray(outcome, dtype=np.float64)
    n_bins = max(4, int(n_bins))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            out.append(
                {
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "count": 0.0,
                    "confidence_mean": 0.0,
                    "accuracy_mean": 0.0,
                    "gap": 0.0,
                }
            )
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        out.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "count": float(mask.sum()),
                "confidence_mean": conf,
                "accuracy_mean": acc,
                "gap": abs(conf - acc),
            }
        )
    return out


def suggest_uncertainty_thresholds(
    uncertainty: np.ndarray,
    outcome: np.ndarray,
    target_error_rates: tuple[float, ...] = (0.05, 0.10, 0.15),
) -> list[dict[str, float]]:
    u = np.clip(np.asarray(uncertainty, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(outcome, dtype=np.float64)
    if u.size == 0:
        return []

    thresholds = np.linspace(0.02, 0.98, 97)
    out: list[dict[str, float]] = []
    for target in target_error_rates:
        best_tau = 1.0
        best_cov = 0.0
        best_err = 1.0
        for tau in thresholds:
            keep = u <= tau
            if not np.any(keep):
                continue
            cov = float(keep.mean())
            err = 1.0 - float(y[keep].mean())
            if err <= target and cov >= best_cov:
                best_cov = cov
                best_tau = float(tau)
                best_err = err
        out.append(
            {
                "target_error_rate": float(target),
                "uncertainty_threshold": float(best_tau),
                "coverage": float(best_cov),
                "expected_error": float(best_err),
            }
        )
    return out
