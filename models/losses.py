"""Loss functions for plume segmentation."""

from __future__ import annotations

import torch
from torch import nn


class DiceBCELoss(nn.Module):
    """Binary cross entropy with Dice overlap penalty."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = 1 - ((2 * intersection + self.smooth) / (union + self.smooth)).mean()
        return bce + dice
