"""Grad-CAM utilities."""

from __future__ import annotations

import torch


def gradient_saliency(model: torch.nn.Module, x: torch.Tensor, target_index: int | None = None) -> torch.Tensor:
    """Compute input-gradient saliency map as a lightweight Grad-CAM starter."""

    model.eval()
    x = x.clone().detach().requires_grad_(True)
    output = model(x)
    target = output.mean() if target_index is None else output.flatten()[target_index]
    target.backward()
    return x.grad.detach().abs().mean(dim=1, keepdim=True)
