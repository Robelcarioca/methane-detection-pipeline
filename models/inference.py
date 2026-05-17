"""Inference helpers."""

from __future__ import annotations

import torch


@torch.inference_mode()
def predict_probability(model: torch.nn.Module, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return methane plume probability map."""

    model.eval()
    logits = model(tensor.to(device))
    return torch.sigmoid(logits).cpu()
