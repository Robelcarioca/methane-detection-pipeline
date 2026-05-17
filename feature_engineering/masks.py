"""Cloud, water, and vegetation mask helpers."""

from __future__ import annotations

import numpy as np


def vegetation_mask(nir: np.ndarray, red: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Create vegetation mask from NDVI."""

    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    return ndvi > threshold


def water_mask(green: np.ndarray, nir: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """Create water mask from NDWI-like contrast."""

    ndwi = (green - nir) / np.maximum(green + nir, 1e-6)
    return ndwi > threshold
