"""Anomaly detection package."""

from anomaly_detection.methane_proxy import (
    MethaneProxyConfig,
    compute_ndvi_deficit_anomaly,
    classify_methane_risk,
    filter_small_clusters_inplace,
    label_stress_clusters,
)

__all__ = [
    "MethaneProxyConfig",
    "compute_ndvi_deficit_anomaly",
    "classify_methane_risk",
    "filter_small_clusters_inplace",
    "label_stress_clusters",
]
