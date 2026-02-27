from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings
from app.model.uncertainty_calibration import apply_temperature_scaling


@dataclass(frozen=True)
class UncertaintyCalibrationProfile:
    model_version: str
    temperature: float
    uncertainty_threshold: float
    uplift_scale: float
    source_path: str
    available: bool
    ece_before: float | None = None
    ece_after: float | None = None
    brier_before: float | None = None
    brier_after: float | None = None


def _default_profile(model_version: str) -> UncertaintyCalibrationProfile:
    return UncertaintyCalibrationProfile(
        model_version=model_version,
        temperature=1.0,
        uncertainty_threshold=0.95,
        uplift_scale=0.0,
        source_path="",
        available=False,
        ece_before=None,
        ece_after=None,
        brier_before=None,
        brier_after=None,
    )


def _normalize_ts_name(name: str) -> str:
    return name.strip().replace(":", "-")


def _extract_threshold_from_suggestions(items: list[dict[str, Any]]) -> tuple[float, float]:
    if not items:
        return 0.65, 0.35
    # Prefer target_error_rate closest to 0.10 as a balanced operating point.
    best = min(items, key=lambda x: abs(float(x.get("target_error_rate", 0.10)) - 0.10))
    tau = float(best.get("uncertainty_threshold", 0.65))
    tau = float(np.clip(tau, 0.05, 0.98))
    # Keep uplift bounded; stronger penalty only for very uncertain cells.
    scale = 0.35
    return tau, scale


def _parse_profile_payload(payload: dict[str, Any], model_version: str, source_path: Path) -> UncertaintyCalibrationProfile:
    profile = payload.get("profile")
    if isinstance(profile, dict):
        return UncertaintyCalibrationProfile(
            model_version=str(profile.get("model_version", model_version)),
            temperature=float(max(1e-4, float(profile.get("temperature", 1.0)))),
            uncertainty_threshold=float(np.clip(float(profile.get("uncertainty_threshold", 0.65)), 0.05, 0.98)),
            uplift_scale=float(max(0.0, float(profile.get("uplift_scale", 0.35)))),
            source_path=str(source_path),
            available=True,
            ece_before=float(profile["ece_before"]) if "ece_before" in profile else None,
            ece_after=float(profile["ece_after"]) if "ece_after" in profile else None,
            brier_before=float(profile["brier_before"]) if "brier_before" in profile else None,
            brier_after=float(profile["brier_after"]) if "brier_after" in profile else None,
        )

    # Backward compatible parsing for report-style payload.
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    suggestions = payload.get("threshold_suggestions") if isinstance(payload.get("threshold_suggestions"), list) else []
    tau, scale = _extract_threshold_from_suggestions(suggestions)
    return UncertaintyCalibrationProfile(
        model_version=str(summary.get("model_version", model_version)),
        temperature=float(max(1e-4, float(summary.get("temperature", 1.0)))),
        uncertainty_threshold=tau,
        uplift_scale=scale,
        source_path=str(source_path),
        available=True,
        ece_before=float(summary["ece_before"]) if "ece_before" in summary else None,
        ece_after=float(summary["ece_after"]) if "ece_after" in summary else None,
        brier_before=float(summary["brier_before"]) if "brier_before" in summary else None,
        brier_after=float(summary["brier_after"]) if "brier_after" in summary else None,
    )


def _candidate_profile_paths(settings: Settings, model_version: str) -> list[Path]:
    model = _normalize_ts_name(model_version)
    root = settings.outputs_root / "calibration"
    return [
        root / model / "calibration.json",
        root / f"{model}_calibration.json",
        root / "calibration.json",
    ]


def load_uncertainty_calibration_profile(settings: Settings, model_version: str) -> UncertaintyCalibrationProfile:
    for path in _candidate_profile_paths(settings, model_version=model_version):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            try:
                return _parse_profile_payload(payload, model_version=model_version, source_path=path)
            except Exception:
                continue
    return _default_profile(model_version=model_version)


def calibrate_uncertainty_grid(uncertainty: np.ndarray, *, temperature: float) -> np.ndarray:
    unc = np.clip(np.nan_to_num(uncertainty.astype(np.float32), nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    confidence = 1.0 - unc
    confidence_cal = apply_temperature_scaling(confidence.astype(np.float64), float(max(1e-4, temperature)))
    unc_cal = 1.0 - confidence_cal
    return np.clip(unc_cal.astype(np.float32), 0.0, 1.0)


def build_uncertainty_penalty_map(
    calibrated_uncertainty: np.ndarray,
    *,
    threshold: float,
    uplift_scale: float,
) -> np.ndarray:
    unc = np.clip(np.nan_to_num(calibrated_uncertainty.astype(np.float32), nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    tau = float(np.clip(threshold, 0.05, 0.98))
    scale = float(max(0.0, uplift_scale))
    if scale <= 1e-8:
        return np.zeros_like(unc, dtype=np.float32)
    rel = np.maximum(0.0, unc - tau) / max(1e-6, 1.0 - tau)
    return np.clip(scale * rel, 0.0, scale).astype(np.float32)

