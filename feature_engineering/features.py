"""Methane-focused feature engineering."""

from __future__ import annotations

import numpy as np


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute a stable band ratio."""

    return (numerator / np.maximum(denominator, eps)).astype(np.float32)


def temporal_difference(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Compute delta reflectance from two matched observations."""

    if previous.shape != current.shape:
        raise ValueError("Temporal arrays must have identical shapes.")
    return (current - previous).astype(np.float32)


def spectral_anomaly(current: np.ndarray, baseline: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute standardized spectral anomaly against a baseline stack."""

    mean = np.nanmean(baseline, axis=0)
    std = np.nanstd(baseline, axis=0)
    return ((current - mean) / np.maximum(std, eps)).astype(np.float32)


def swir_methane_indices(stack: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute methane-sensitive SWIR starter indices."""

    features: dict[str, np.ndarray] = {}
    if "B11" in stack and "B12" in stack:
        features["swir_ratio_b12_b11"] = safe_ratio(stack["B12"], stack["B11"])
        features["swir_delta_b12_b11"] = (stack["B12"] - stack["B11"]).astype(np.float32)
    return features
