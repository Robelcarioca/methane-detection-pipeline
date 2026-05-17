"""Vegetation index and stress classification utilities."""

from __future__ import annotations

import numpy as np


def compute_ndvi(red: np.ndarray, nir: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute NDVI from red and near-infrared reflectance arrays."""

    red32 = red.astype(np.float32, copy=False)
    nir32 = nir.astype(np.float32, copy=False)
    denominator = nir32 + red32
    ndvi = np.full(red32.shape, np.nan, dtype=np.float32)
    valid = np.abs(denominator) > eps
    ndvi[valid] = (nir32[valid] - red32[valid]) / denominator[valid]
    return ndvi


def classify_vegetation_stress(
    ndvi: np.ndarray,
    stress_threshold: float = 0.2,
    healthy_threshold: float = 0.5,
    nodata_value: int = 255,
) -> np.ndarray:
    """Classify NDVI into stress classes.

    Classes:
    - 0: stressed or non-vegetated
    - 1: moderate vegetation
    - 2: healthy vegetation
    - 255: nodata
    """

    stress = np.full(ndvi.shape, nodata_value, dtype=np.uint8)
    valid = np.isfinite(ndvi)
    stress[valid & (ndvi < stress_threshold)] = 0
    stress[valid & (ndvi >= stress_threshold) & (ndvi <= healthy_threshold)] = 1
    stress[valid & (ndvi > healthy_threshold)] = 2
    return stress
