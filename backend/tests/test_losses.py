from __future__ import annotations

import torch

from app.model.losses import FocalDiceLoss


def test_focal_dice_loss_backward_and_finite() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    target = torch.randint(0, 3, (2, 16, 16), dtype=torch.long)
    class_weights = torch.tensor([0.8, 1.4, 1.8], dtype=torch.float32)

    criterion = FocalDiceLoss(
        num_classes=3,
        class_weights=class_weights,
        focal_gamma=2.0,
        dice_smooth=1.0,
        lambda_ce=0.4,
        lambda_focal=0.3,
        lambda_dice=0.3,
    )
    loss = criterion(logits, target)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all().item()
