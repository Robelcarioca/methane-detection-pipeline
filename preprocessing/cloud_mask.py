"""Cloud masking utilities."""

from __future__ import annotations

import numpy as np


def simple_cloud_mask(rgb: np.ndarray, brightness_threshold: float = 0.85) -> np.ndarray:
    """Create a simple brightness-based cloud mask from RGB-like channels.

    Args:
        rgb: Array shaped `[3, height, width]` with reflectance in `[0, 1]`.
    """

    if rgb.shape[0] != 3:
        raise ValueError("Expected RGB array with shape [3, height, width].")
    brightness = rgb.mean(axis=0)
    return brightness > brightness_threshold
