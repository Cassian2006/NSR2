from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LossPreset:
    focal_gamma: float
    dice_smooth: float
    lambda_ce: float
    lambda_focal: float
    lambda_dice: float


LOSS_PRESETS: dict[str, LossPreset] = {
    "none": LossPreset(
        focal_gamma=2.0,
        dice_smooth=1.0,
        lambda_ce=0.4,
        lambda_focal=0.3,
        lambda_dice=0.3,
    ),
    "balanced": LossPreset(
        focal_gamma=2.0,
        dice_smooth=1.0,
        lambda_ce=0.4,
        lambda_focal=0.3,
        lambda_dice=0.3,
    ),
    "caution_focus": LossPreset(
        focal_gamma=2.4,
        dice_smooth=1.0,
        lambda_ce=0.25,
        lambda_focal=0.40,
        lambda_dice=0.35,
    ),
    "blocked_focus": LossPreset(
        focal_gamma=2.2,
        dice_smooth=1.0,
        lambda_ce=0.30,
        lambda_focal=0.35,
        lambda_dice=0.35,
    ),
}


def parse_class_weights(raw: str) -> np.ndarray:
    txt = (raw or "").strip()
    if not txt:
        raise ValueError("class weights are empty")
    parts = [p.strip() for p in txt.split(",")]
    if len(parts) != 3:
        raise ValueError("class weights must contain exactly 3 comma-separated values")
    vals = np.asarray([float(x) for x in parts], dtype=np.float32)
    if not np.isfinite(vals).all():
        raise ValueError("class weights must be finite")
    if (vals <= 0).any():
        raise ValueError("class weights must be > 0")
    return vals


def resolve_class_weights(
    *,
    mode: str,
    auto_weights: np.ndarray,
    manual_raw: str = "",
) -> tuple[np.ndarray, str]:
    if mode == "uniform":
        return np.ones(3, dtype=np.float32), "uniform"
    if mode == "manual":
        return parse_class_weights(manual_raw), "manual"
    return np.asarray(auto_weights, dtype=np.float32), "auto"


def resolve_loss_hparams(
    *,
    loss: str,
    preset: str,
    focal_gamma: float,
    dice_smooth: float,
    lambda_ce: float,
    lambda_focal: float,
    lambda_dice: float,
) -> dict[str, float | str]:
    if loss != "focal_dice":
        return {
            "loss": loss,
            "loss_preset": "none",
            "focal_gamma": float(focal_gamma),
            "dice_smooth": float(dice_smooth),
            "lambda_ce": float(lambda_ce),
            "lambda_focal": float(lambda_focal),
            "lambda_dice": float(lambda_dice),
        }

    if preset not in LOSS_PRESETS:
        raise ValueError(f"unknown loss preset: {preset}")

    if preset == "none":
        return {
            "loss": loss,
            "loss_preset": preset,
            "focal_gamma": float(focal_gamma),
            "dice_smooth": float(dice_smooth),
            "lambda_ce": float(lambda_ce),
            "lambda_focal": float(lambda_focal),
            "lambda_dice": float(lambda_dice),
        }

    cfg = LOSS_PRESETS[preset]
    return {
        "loss": loss,
        "loss_preset": preset,
        "focal_gamma": float(cfg.focal_gamma),
        "dice_smooth": float(cfg.dice_smooth),
        "lambda_ce": float(cfg.lambda_ce),
        "lambda_focal": float(cfg.lambda_focal),
        "lambda_dice": float(cfg.lambda_dice),
    }
