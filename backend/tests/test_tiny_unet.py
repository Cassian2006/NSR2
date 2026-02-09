from __future__ import annotations

import torch

from app.model.tiny_unet import TinyUNet


def test_tiny_unet_forward_shape() -> None:
    model = TinyUNet(in_channels=7, n_classes=3, base=16)
    x = torch.randn(2, 7, 128, 192)
    y = model(x)
    assert y.shape == (2, 3, 128, 192)
