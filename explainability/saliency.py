"""Saliency map helpers."""

from __future__ import annotations

import torch


def input_saliency(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Compute absolute input-gradient saliency."""

    x = x.clone().detach().requires_grad_(True)
    score = model(x).mean()
    score.backward()
    return x.grad.abs()
