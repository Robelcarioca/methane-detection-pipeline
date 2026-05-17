"""Band alignment, resampling, reprojection, and normalization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def normalize_reflectance(
    array: np.ndarray,
    lower_percentile: float = 2,
    upper_percentile: float = 98,
    eps: float = 1e-6,
) -> np.ndarray:
    """Robust percentile normalization for reflectance arrays."""

    lower = np.nanpercentile(array, lower_percentile, axis=(-2, -1), keepdims=True)
    upper = np.nanpercentile(array, upper_percentile, axis=(-2, -1), keepdims=True)
    normalized = (array - lower) / np.maximum(upper - lower, eps)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def fill_missing(array: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Replace NaN and infinite values."""

    return np.nan_to_num(array, nan=fill_value, posinf=fill_value, neginf=fill_value).astype(np.float32)


def align_and_resample_bands(
    band_paths: dict[str, Path],
    target_resolution_m: int = 20,
) -> np.ndarray:
    """Load and align bands to a common grid.

    This placeholder returns zeros when no real imagery exists. Production code
    should use rasterio/rioxarray to reproject each band to the reference grid.
    """

    _ = target_resolution_m
    channel_count = max(len(band_paths), 1)
    return np.zeros((channel_count, 256, 256), dtype=np.float32)
