from __future__ import annotations

import numpy as np
import pytest

from app.model.train_config import parse_class_weights, resolve_class_weights, resolve_loss_hparams


def test_parse_class_weights_ok() -> None:
    vals = parse_class_weights("0.8,1.4,1.8")
    assert vals.shape == (3,)
    assert np.allclose(vals, np.asarray([0.8, 1.4, 1.8], dtype=np.float32))


@pytest.mark.parametrize("raw", ["", "1,2", "1,2,0", "a,2,3"])
def test_parse_class_weights_invalid(raw: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        parse_class_weights(raw)


def test_resolve_class_weights_modes() -> None:
    auto = np.asarray([0.7, 1.5, 1.9], dtype=np.float32)
    vals_auto, source_auto = resolve_class_weights(mode="auto", auto_weights=auto)
    vals_uniform, source_uniform = resolve_class_weights(mode="uniform", auto_weights=auto)
    vals_manual, source_manual = resolve_class_weights(
        mode="manual",
        auto_weights=auto,
        manual_raw="0.9,1.2,1.7",
    )

    assert source_auto == "auto"
    assert np.allclose(vals_auto, auto)
    assert source_uniform == "uniform"
    assert np.allclose(vals_uniform, np.ones(3, dtype=np.float32))
    assert source_manual == "manual"
    assert np.allclose(vals_manual, np.asarray([0.9, 1.2, 1.7], dtype=np.float32))


def test_resolve_loss_hparams_applies_preset_for_focal_dice() -> None:
    cfg = resolve_loss_hparams(
        loss="focal_dice",
        preset="caution_focus",
        focal_gamma=1.0,
        dice_smooth=2.0,
        lambda_ce=0.1,
        lambda_focal=0.1,
        lambda_dice=0.8,
    )
    assert cfg["loss"] == "focal_dice"
    assert cfg["loss_preset"] == "caution_focus"
    assert cfg["focal_gamma"] == pytest.approx(2.4)
    assert cfg["lambda_ce"] == pytest.approx(0.25)


def test_resolve_loss_hparams_keeps_cli_values_when_none() -> None:
    cfg = resolve_loss_hparams(
        loss="focal_dice",
        preset="none",
        focal_gamma=1.3,
        dice_smooth=1.5,
        lambda_ce=0.2,
        lambda_focal=0.4,
        lambda_dice=0.4,
    )
    assert cfg["loss_preset"] == "none"
    assert cfg["focal_gamma"] == pytest.approx(1.3)
    assert cfg["lambda_ce"] == pytest.approx(0.2)


def test_resolve_loss_hparams_ignores_preset_for_ce() -> None:
    cfg = resolve_loss_hparams(
        loss="ce",
        preset="blocked_focus",
        focal_gamma=2.1,
        dice_smooth=1.1,
        lambda_ce=0.6,
        lambda_focal=0.2,
        lambda_dice=0.2,
    )
    assert cfg["loss"] == "ce"
    assert cfg["loss_preset"] == "none"
    assert cfg["lambda_ce"] == pytest.approx(0.6)
