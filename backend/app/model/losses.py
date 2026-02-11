from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceLoss(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
        focal_gamma: float = 2.0,
        focal_alpha: torch.Tensor | None = None,
        dice_smooth: float = 1.0,
        lambda_ce: float = 0.4,
        lambda_focal: float = 0.3,
        lambda_dice: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.focal_gamma = float(max(0.0, focal_gamma))
        self.dice_smooth = float(max(1e-6, dice_smooth))
        self.lambda_ce = float(max(0.0, lambda_ce))
        self.lambda_focal = float(max(0.0, lambda_focal))
        self.lambda_dice = float(max(0.0, lambda_dice))
        w_sum = self.lambda_ce + self.lambda_focal + self.lambda_dice
        if w_sum <= 0:
            self.lambda_ce = 1.0
            self.lambda_focal = 0.0
            self.lambda_dice = 0.0
        else:
            self.lambda_ce /= w_sum
            self.lambda_focal /= w_sum
            self.lambda_dice /= w_sum

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None  # type: ignore[assignment]

        if focal_alpha is not None:
            self.register_buffer("focal_alpha", focal_alpha.float())
        else:
            self.focal_alpha = None  # type: ignore[assignment]

    def _focal_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_idx = target.unsqueeze(1)
        pt = probs.gather(1, target_idx).squeeze(1).clamp_min(1e-8)
        log_pt = log_probs.gather(1, target_idx).squeeze(1)

        modulating = (1.0 - pt).pow(self.focal_gamma)
        if self.focal_alpha is not None:
            alpha_t = self.focal_alpha.gather(0, target.reshape(-1)).reshape_as(target).to(logits.dtype)
        else:
            alpha_t = torch.ones_like(pt)
        return (-alpha_t * modulating * log_pt).mean()

    def _dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.dice_smooth) / (denom + self.dice_smooth)
        if self.class_weights is not None:
            w = self.class_weights / self.class_weights.sum().clamp_min(1e-8)
            return 1.0 - (dice * w).sum()
        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.class_weights)
        focal = self._focal_loss(logits, target)
        dice = self._dice_loss(logits, target)
        return self.lambda_ce * ce + self.lambda_focal * focal + self.lambda_dice * dice
