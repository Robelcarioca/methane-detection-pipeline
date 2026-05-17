"""Simple CNN baseline for plume segmentation."""

from __future__ import annotations

import torch
from torch import nn


class CNNBaseline(nn.Module):
    """Small convolutional segmentation baseline."""

    def __init__(self, in_channels: int = 20, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
