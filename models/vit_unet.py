"""ViT encoder plus U-Net-style decoder starter."""

from __future__ import annotations

import torch
from torch import nn

from models.unet import UNet


class ViTUNet(nn.Module):
    """Hybrid model boundary for transformer/U-Net experiments.

    The starter keeps a U-Net decoder contract while providing a clean place to
    plug in `timm` feature extractors.
    """

    def __init__(
        self,
        in_channels: int = 20,
        out_channels: int = 1,
        base_channels: int = 32,
        encoder_name: str = "vit_base_patch16_224",
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.backbone = UNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
