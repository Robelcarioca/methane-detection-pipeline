from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from anomaly_detection.methane_proxy import (
    MethaneProxyConfig,
    classify_methane_risk,
    compute_ndvi_deficit_anomaly,
    label_stress_clusters,
)
from feature_engineering.vegetation import classify_vegetation_stress, compute_ndvi
from preprocessing.sentinel2_safe import resolve_r10m_vegetation_bands
from preprocessing.vegetation_pipeline import process_sentinel2_scene


def _artifact_dir() -> Path:
    path = Path("manual_tmp") / "test_artifacts" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_test_raster(path: Path, array: np.ndarray) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": str(array.dtype),
        "crs": "EPSG:32632",
        "transform": from_origin(500000, 10000, 10, 10),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def test_compute_ndvi_and_stress_classes() -> None:
    red = np.array([[0.2, 0.4, 0.0]], dtype=np.float32)
    nir = np.array([[0.8, 0.6, 0.0]], dtype=np.float32)

    ndvi = compute_ndvi(red=red, nir=nir)
    stress = classify_vegetation_stress(ndvi)

    np.testing.assert_allclose(ndvi[0, :2], np.array([0.6, 0.2], dtype=np.float32))
    assert np.isnan(ndvi[0, 2])
    assert stress.tolist() == [[2, 1, 255]]


def test_methane_proxy_anomaly_risk_and_clusters() -> None:
    local_ndvi = np.array([[0.6, 0.2, np.nan], [0.1, 0.15, 0.7]], dtype=np.float32)
    stress = np.array([[2, 1, 255], [0, 0, 2]], dtype=np.uint8)

    anomaly = compute_ndvi_deficit_anomaly(local_ndvi, global_mean=0.5, global_std=0.1)
    risk = classify_methane_risk(anomaly, stress, medium_anomaly_z=1.0, high_anomaly_z=2.0)
    clusters, count = label_stress_clusters((risk >= 1).astype(np.uint8))

    assert risk.tolist() == [[0, 1, 0], [2, 2, 0]]
    assert count == 1
    assert int(clusters[0, 1]) == int(clusters[1, 0]) == int(clusters[1, 1])


def test_resolve_r10m_bands_and_process_scene() -> None:
    pytest.importorskip("rasterio")
    root = _artifact_dir()
    safe = root / "S2_TEST.SAFE"
    img_data = safe / "GRANULE" / "L2A_TEST" / "IMG_DATA" / "R10m"
    red_path = img_data / "T32TEST_20260513T094029_B04_10m.jp2"
    nir_path = img_data / "T32TEST_20260513T094029_B08_10m.jp2"

    red = np.full((16, 16), 2000, dtype=np.uint16)
    nir = np.full((16, 16), 8000, dtype=np.uint16)
    _write_test_raster(red_path, red)
    _write_test_raster(nir_path, nir)

    bands = resolve_r10m_vegetation_bands(safe)
    outputs = process_sentinel2_scene(
        bands,
        root / "outputs",
        max_preview_size=8,
        methane_proxy=MethaneProxyConfig(smoothing_sigma=1.0, min_cluster_pixels=1),
    )

    assert outputs.ndvi_tif.exists()
    assert outputs.stress_map_tif.exists()
    assert outputs.methane_risk_map_tif.exists()
    assert outputs.anomaly_map_npy.exists()
    assert outputs.cluster_map_npy.exists()
    assert outputs.ndvi_png.exists()
    assert outputs.stress_map_png.exists()
    assert outputs.methane_risk_png.exists()
    assert outputs.ndvi_npy.exists()
    assert outputs.stress_map_npy.exists()

    ndvi = np.load(outputs.ndvi_npy)
    stress = np.load(outputs.stress_map_npy)
    anomaly = np.load(outputs.anomaly_map_npy)
    clusters = np.load(outputs.cluster_map_npy)
    assert ndvi.dtype == np.float32
    assert stress.dtype == np.uint8
    assert anomaly.dtype == np.float32
    assert clusters.dtype == np.int32
    np.testing.assert_allclose(ndvi.mean(), 0.6, rtol=1e-5)
    assert int(stress.max()) == 2
    assert np.isfinite(anomaly).all()
    assert int(clusters.max()) == 0

    flat_outputs = process_sentinel2_scene(
        bands,
        root / "final",
        max_preview_size=8,
        flat_output=True,
        methane_proxy=MethaneProxyConfig(smoothing_sigma=1.0, min_cluster_pixels=1),
    )
    assert flat_outputs.output_dir == root / "final"
    assert (root / "final" / "ndvi.tif").exists()
    assert (root / "final" / "methane_risk_map.tif").exists()
