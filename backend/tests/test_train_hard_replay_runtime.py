from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_unet_quick import run_epoch


class _TinyHead(nn.Module):
    def __init__(self, in_ch: int = 4, n_cls: int = 3) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_ch, n_cls, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def test_run_epoch_reports_hard_hit_rate_from_loader_flags() -> None:
    torch.manual_seed(0)
    xb = torch.randn(4, 4, 8, 8)
    yb = torch.randint(0, 3, (4, 8, 8), dtype=torch.long)
    hb = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float32)
    loader = DataLoader(TensorDataset(xb, yb, hb), batch_size=2, shuffle=False)

    model = _TinyHead(in_ch=4, n_cls=3)
    criterion = nn.CrossEntropyLoss()
    metrics = run_epoch(model, loader, criterion, device=torch.device("cpu"), optimizer=None)

    assert "hard_hit_rate" in metrics
    assert abs(float(metrics["hard_hit_rate"]) - 0.75) < 1e-6
