"""Transformer attention visualization helpers."""

from __future__ import annotations

import torch


def normalize_attention(attention: torch.Tensor) -> torch.Tensor:
    """Normalize attention maps to `[0, 1]` for visualization."""

    attention = attention.detach().float()
    return (attention - attention.min()) / torch.clamp(attention.max() - attention.min(), min=1e-6)
