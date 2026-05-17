"""Methane leakage proxy features derived from Sentinel-2 vegetation stress."""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableSequence

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class MethaneProxyConfig:
    """Configuration for NDVI-anomaly methane proxy products.

    The anomaly convention is positive for NDVI deficit:
    ``(global_mean - local_smoothed_ndvi) / global_std``.
    """

    smoothing_sigma: float = 3.0
    medium_anomaly_z: float = 1.0
    high_anomaly_z: float = 1.75
    min_cluster_pixels: int = 9
    nodata_value: int = 255

    @property
    def gaussian_halo(self) -> int:
        """Halo radius required for block-wise Gaussian smoothing."""

        return max(1, int(np.ceil(self.smoothing_sigma * 4)))


def compute_ndvi_deficit_anomaly(
    local_ndvi: np.ndarray,
    global_mean: float,
    global_std: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Return positive z-score anomalies where local NDVI is below scene mean."""

    ndvi32 = local_ndvi.astype(np.float32, copy=False)
    anomaly = np.full(ndvi32.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(ndvi32)
    scale = max(float(global_std), eps)
    anomaly[valid] = (float(global_mean) - ndvi32[valid]) / scale
    return anomaly


def classify_methane_risk(
    anomaly: np.ndarray,
    stress_map: np.ndarray,
    medium_anomaly_z: float = 1.0,
    high_anomaly_z: float = 1.75,
    nodata_value: int = 255,
) -> np.ndarray:
    """Classify methane proxy risk from anomaly and vegetation stress maps.

    Classes:
    - 0: low risk
    - 1: medium risk
    - 2: high risk
    """

    if anomaly.shape != stress_map.shape:
        raise ValueError(f"Shape mismatch: anomaly={anomaly.shape}, stress_map={stress_map.shape}")

    risk = np.zeros(anomaly.shape, dtype=np.uint8)
    valid = np.isfinite(anomaly) & (stress_map != nodata_value)
    stressed_or_moderate = valid & (stress_map <= 1)
    stressed = valid & (stress_map == 0)

    risk[stressed_or_moderate & (anomaly >= medium_anomaly_z)] = 1
    risk[stressed & (anomaly >= high_anomaly_z)] = 2
    return risk


def label_stress_clusters(seed_map: np.ndarray, output: np.ndarray | None = None) -> tuple[np.ndarray, int] | int:
    """Label contiguous nonzero stress/risk zones with 8-neighbor connectivity."""

    structure = np.ones((3, 3), dtype=np.uint8)
    if output is None:
        return ndi.label(seed_map, structure=structure)
    return int(ndi.label(seed_map, structure=structure, output=output))


def filter_small_clusters_inplace(
    cluster_map: np.ndarray,
    num_clusters: int,
    min_cluster_pixels: int,
    rows_per_chunk: int = 1024,
    linked_risk_map: np.ndarray | None = None,
) -> int:
    """Remove small connected components from a label map and optional risk map."""

    if num_clusters <= 0:
        return 0
    if min_cluster_pixels <= 1:
        return num_clusters

    counts = np.zeros(num_clusters + 1, dtype=np.uint64)
    height = cluster_map.shape[0]
    for row_start in range(0, height, rows_per_chunk):
        row_stop = min(row_start + rows_per_chunk, height)
        chunk = np.asarray(cluster_map[row_start:row_stop])
        counts += np.bincount(chunk.ravel(), minlength=num_clusters + 1).astype(np.uint64, copy=False)

    remove = counts < int(min_cluster_pixels)
    remove[0] = False
    kept = int(np.count_nonzero(~remove[1:]))

    for row_start in range(0, height, rows_per_chunk):
        row_stop = min(row_start + rows_per_chunk, height)
        labels = cluster_map[row_start:row_stop]
        small = remove[labels]
        if not np.any(small):
            continue
        labels[small] = 0
        if linked_risk_map is not None:
            risk = linked_risk_map[row_start:row_stop]
            risk[small] = 0

    for mmap in _flushable_arrays(cluster_map, linked_risk_map):
        mmap.flush()
    return kept


def _flushable_arrays(*arrays: np.ndarray | None) -> MutableSequence[np.memmap]:
    return [array for array in arrays if isinstance(array, np.memmap)]
