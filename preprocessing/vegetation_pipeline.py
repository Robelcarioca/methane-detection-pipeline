"""Windowed Sentinel-2 NDVI, stress, anomaly, and methane proxy pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from numpy.lib.format import open_memmap
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter

from anomaly_detection.methane_proxy import (
    MethaneProxyConfig,
    classify_methane_risk,
    compute_ndvi_deficit_anomaly,
    filter_small_clusters_inplace,
    label_stress_clusters,
)
from data_ingestion.sentinel2_safe import Sentinel2BandSet, discover_safe_products, resolve_r10m_vegetation_bands
from feature_engineering.vegetation import classify_vegetation_stress, compute_ndvi
from visualization.vegetation import save_methane_risk_png, save_ndvi_png, save_stress_png


@dataclass(frozen=True)
class VegetationPipelineConfig:
    """Runtime configuration for Sentinel-2 vegetation proxy processing."""

    raw_data: Path
    output_dir: Path
    reflectance_scale: float = 10000.0
    max_preview_size: int = 2048
    overwrite: bool = True
    flat_output: bool = False
    methane_proxy: MethaneProxyConfig = field(default_factory=MethaneProxyConfig)


@dataclass(frozen=True)
class SceneOutputs:
    """Output file paths for a processed scene."""

    scene_id: str
    output_dir: Path
    ndvi_tif: Path
    stress_map_tif: Path
    methane_risk_map_tif: Path
    anomaly_map_npy: Path
    cluster_map_npy: Path
    ndvi_png: Path
    stress_map_png: Path
    methane_risk_png: Path
    ndvi_npy: Path
    stress_map_npy: Path


@dataclass(frozen=True)
class RasterStats:
    """Streaming raster statistics for valid NDVI pixels."""

    count: int
    mean: float
    std: float


def _scene_outputs(scene_id: str, output_root: Path, flat_output: bool = False) -> SceneOutputs:
    scene_dir = output_root if flat_output else output_root / scene_id
    return SceneOutputs(
        scene_id=scene_id,
        output_dir=scene_dir,
        ndvi_tif=scene_dir / "ndvi.tif",
        stress_map_tif=scene_dir / "stress_map.tif",
        methane_risk_map_tif=scene_dir / "methane_risk_map.tif",
        anomaly_map_npy=scene_dir / "anomaly_map.npy",
        cluster_map_npy=scene_dir / "cluster_map.npy",
        ndvi_png=scene_dir / "ndvi.png",
        stress_map_png=scene_dir / "stress_map.png",
        methane_risk_png=scene_dir / "methane_risk.png",
        ndvi_npy=scene_dir / "ndvi.npy",
        stress_map_npy=scene_dir / "stress_map.npy",
    )


def _prepare_profile(src: rasterio.io.DatasetReader, dtype: str, nodata: float | int | None) -> dict:
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    if nodata is None:
        profile.pop("nodata", None)
    else:
        profile["nodata"] = nodata
    return profile


def _preview_shape(height: int, width: int, max_size: int) -> tuple[int, int]:
    scale = max(height / max_size, width / max_size, 1.0)
    return max(1, int(round(height / scale))), max(1, int(round(width / scale)))


def _write_previews(outputs: SceneOutputs, max_preview_size: int) -> None:
    with rasterio.open(outputs.ndvi_tif) as ndvi_src:
        out_height, out_width = _preview_shape(ndvi_src.height, ndvi_src.width, max_preview_size)
        ndvi_preview = ndvi_src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.average,
            masked=True,
        ).filled(np.nan)
    save_ndvi_png(ndvi_preview, outputs.ndvi_png)

    with rasterio.open(outputs.stress_map_tif) as stress_src:
        out_height, out_width = _preview_shape(stress_src.height, stress_src.width, max_preview_size)
        stress_preview = stress_src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.nearest,
            masked=True,
        ).filled(255)
    save_stress_png(stress_preview.astype(np.uint8), outputs.stress_map_png)

    with rasterio.open(outputs.methane_risk_map_tif) as risk_src:
        out_height, out_width = _preview_shape(risk_src.height, risk_src.width, max_preview_size)
        risk_preview = risk_src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.nearest,
            masked=True,
        ).filled(0)
    save_methane_risk_png(risk_preview.astype(np.uint8), outputs.methane_risk_png)


def _window_slices(window: Window) -> tuple[slice, slice]:
    row_start = int(window.row_off)
    row_stop = row_start + int(window.height)
    col_start = int(window.col_off)
    col_stop = col_start + int(window.width)
    return slice(row_start, row_stop), slice(col_start, col_stop)


def _expanded_window(window: Window, height: int, width: int, halo: int) -> tuple[Window, tuple[slice, slice]]:
    row_start = int(window.row_off)
    row_stop = row_start + int(window.height)
    col_start = int(window.col_off)
    col_stop = col_start + int(window.width)

    expanded_row_start = max(0, row_start - halo)
    expanded_row_stop = min(height, row_stop + halo)
    expanded_col_start = max(0, col_start - halo)
    expanded_col_stop = min(width, col_stop + halo)

    expanded = Window(
        col_off=expanded_col_start,
        row_off=expanded_row_start,
        width=expanded_col_stop - expanded_col_start,
        height=expanded_row_stop - expanded_row_start,
    )
    crop = (
        slice(row_start - expanded_row_start, row_stop - expanded_row_start),
        slice(col_start - expanded_col_start, col_stop - expanded_col_start),
    )
    return expanded, crop


def _update_stats(count: int, total: float, total_sq: float, array: np.ndarray) -> tuple[int, float, float]:
    valid_values = array[np.isfinite(array)].astype(np.float64, copy=False)
    if valid_values.size == 0:
        return count, total, total_sq
    return (
        count + int(valid_values.size),
        total + float(valid_values.sum(dtype=np.float64)),
        total_sq + float(np.square(valid_values, dtype=np.float64).sum(dtype=np.float64)),
    )


def _finalize_stats(count: int, total: float, total_sq: float) -> RasterStats:
    if count <= 0:
        raise ValueError("No valid NDVI pixels were produced; cannot compute anomaly map.")
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    std = max(math.sqrt(variance), 1e-6)
    return RasterStats(count=count, mean=mean, std=std)


def _write_risk_geotiff_from_memmap(
    risk_map: np.ndarray,
    output_path: Path,
    reference_src: rasterio.io.DatasetReader,
) -> None:
    risk_profile = _prepare_profile(reference_src, "uint8", 255)
    with rasterio.open(output_path, "w", **risk_profile) as risk_dst:
        for _, window in reference_src.block_windows(1):
            rows, cols = _window_slices(window)
            risk_dst.write(np.asarray(risk_map[rows, cols], dtype=np.uint8), 1, window=window)


def _write_methane_proxy_products(
    outputs: SceneOutputs,
    stats: RasterStats,
    config: MethaneProxyConfig,
) -> None:
    risk_tmp = outputs.output_dir / "_methane_risk_map.tmp.npy"
    seed_tmp = outputs.output_dir / "_cluster_seed.tmp.npy"
    risk_mm: np.memmap | None = None
    seed_mm: np.memmap | None = None
    cluster_mm: np.memmap | None = None

    try:
        with rasterio.open(outputs.ndvi_tif) as ndvi_src, rasterio.open(outputs.stress_map_tif) as stress_src:
            if ndvi_src.shape != stress_src.shape:
                raise ValueError(f"NDVI/stress shape mismatch: {ndvi_src.shape} != {stress_src.shape}")
            if ndvi_src.crs != stress_src.crs or ndvi_src.transform != stress_src.transform:
                raise ValueError("NDVI and stress maps are not on the same grid.")

            shape = ndvi_src.shape
            anomaly_mm = open_memmap(outputs.anomaly_map_npy, mode="w+", dtype=np.float32, shape=shape)
            risk_mm = open_memmap(risk_tmp, mode="w+", dtype=np.uint8, shape=shape)
            seed_mm = open_memmap(seed_tmp, mode="w+", dtype=np.uint8, shape=shape)

            for _, window in ndvi_src.block_windows(1):
                expanded, crop = _expanded_window(window, ndvi_src.height, ndvi_src.width, config.gaussian_halo)
                ndvi_tile = ndvi_src.read(1, window=expanded).astype(np.float32, copy=False)
                ndvi_tile = np.where(np.isfinite(ndvi_tile), ndvi_tile, stats.mean).astype(np.float32, copy=False)
                if config.smoothing_sigma > 0:
                    smoothed = gaussian_filter(ndvi_tile, sigma=config.smoothing_sigma, mode="nearest")
                else:
                    smoothed = ndvi_tile

                local_ndvi = smoothed[crop].astype(np.float32, copy=False)
                center_ndvi = ndvi_src.read(1, window=window).astype(np.float32, copy=False)
                anomaly = compute_ndvi_deficit_anomaly(local_ndvi, stats.mean, stats.std)
                anomaly[~np.isfinite(center_ndvi)] = np.nan

                stress = stress_src.read(1, window=window).astype(np.uint8, copy=False)
                risk = classify_methane_risk(
                    anomaly,
                    stress,
                    medium_anomaly_z=config.medium_anomaly_z,
                    high_anomaly_z=config.high_anomaly_z,
                    nodata_value=config.nodata_value,
                )

                rows, cols = _window_slices(window)
                anomaly_mm[rows, cols] = anomaly
                risk_mm[rows, cols] = risk
                seed_mm[rows, cols] = (risk >= 1).astype(np.uint8)

            anomaly_mm.flush()
            risk_mm.flush()
            seed_mm.flush()
            del anomaly_mm

            cluster_mm = open_memmap(outputs.cluster_map_npy, mode="w+", dtype=np.int32, shape=shape)
            num_clusters = label_stress_clusters(seed_mm, output=cluster_mm)
            seed_mm.flush()
            del seed_mm
            seed_mm = None

            filter_small_clusters_inplace(
                cluster_mm,
                num_clusters=num_clusters,
                min_cluster_pixels=config.min_cluster_pixels,
                linked_risk_map=risk_mm,
            )
            cluster_mm.flush()
            del cluster_mm
            cluster_mm = None

            risk_mm.flush()
            _write_risk_geotiff_from_memmap(risk_mm, outputs.methane_risk_map_tif, ndvi_src)
            del risk_mm
            risk_mm = None
    finally:
        for mmap in (risk_mm, seed_mm, cluster_mm):
            if isinstance(mmap, np.memmap):
                mmap.flush()
        for tmp_path in (risk_tmp, seed_tmp):
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def process_sentinel2_scene(
    bands: Sentinel2BandSet,
    output_root: str | Path,
    reflectance_scale: float = 10000.0,
    max_preview_size: int = 2048,
    overwrite: bool = True,
    flat_output: bool = False,
    methane_proxy: MethaneProxyConfig | None = None,
) -> SceneOutputs:
    """Process one Sentinel-2 SAFE scene into final vegetation proxy products.

    Reads and writes raster windows so the 10 m scene is never materialized as a
    full in-memory float array. NumPy outputs are backed by ``.npy`` memmaps
    while they are being written.
    """

    proxy_config = methane_proxy or MethaneProxyConfig()
    outputs = _scene_outputs(bands.scene_id, Path(output_root), flat_output=flat_output)
    outputs.output_dir.mkdir(parents=True, exist_ok=True)

    required_data = (
        outputs.ndvi_tif,
        outputs.stress_map_tif,
        outputs.methane_risk_map_tif,
        outputs.anomaly_map_npy,
        outputs.cluster_map_npy,
        outputs.ndvi_npy,
        outputs.stress_map_npy,
    )
    if not overwrite and all(path.exists() for path in required_data):
        _write_previews(outputs, max_preview_size)
        return outputs

    with rasterio.open(bands.red) as red_src, rasterio.open(bands.nir) as nir_src:
        if red_src.shape != nir_src.shape:
            raise ValueError(f"Band shape mismatch for {bands.safe_path}: B04={red_src.shape}, B08={nir_src.shape}")
        if red_src.crs != nir_src.crs or red_src.transform != nir_src.transform:
            raise ValueError(f"B04 and B08 are not on the same grid for {bands.safe_path}")

        ndvi_mm = open_memmap(outputs.ndvi_npy, mode="w+", dtype=np.float32, shape=red_src.shape)
        stress_mm = open_memmap(outputs.stress_map_npy, mode="w+", dtype=np.uint8, shape=red_src.shape)

        ndvi_profile = _prepare_profile(red_src, "float32", np.nan)
        stress_profile = _prepare_profile(red_src, "uint8", 255)
        valid_count = 0
        ndvi_total = 0.0
        ndvi_total_sq = 0.0

        with rasterio.open(outputs.ndvi_tif, "w", **ndvi_profile) as ndvi_dst, rasterio.open(
            outputs.stress_map_tif, "w", **stress_profile
        ) as stress_dst:
            for _, window in red_src.block_windows(1):
                red = red_src.read(1, window=window).astype(np.float32) / reflectance_scale
                nir = nir_src.read(1, window=window).astype(np.float32) / reflectance_scale

                ndvi = compute_ndvi(red=red, nir=nir)
                stress = classify_vegetation_stress(ndvi)
                valid_count, ndvi_total, ndvi_total_sq = _update_stats(valid_count, ndvi_total, ndvi_total_sq, ndvi)

                rows, cols = _window_slices(window)
                ndvi_mm[rows, cols] = ndvi
                stress_mm[rows, cols] = stress
                ndvi_dst.write(ndvi, 1, window=window)
                stress_dst.write(stress, 1, window=window)

        ndvi_mm.flush()
        stress_mm.flush()
        del ndvi_mm
        del stress_mm

    ndvi_stats = _finalize_stats(valid_count, ndvi_total, ndvi_total_sq)
    _write_methane_proxy_products(outputs, ndvi_stats, proxy_config)
    _write_previews(outputs, max_preview_size)
    return outputs


def run_vegetation_batch(config: VegetationPipelineConfig) -> list[SceneOutputs]:
    """Process every Sentinel-2 SAFE product under the configured raw folder."""

    products = discover_safe_products(config.raw_data)
    outputs: list[SceneOutputs] = []
    failures: list[str] = []
    flat_output = config.flat_output and len(products) == 1

    for safe_path in products:
        try:
            bands = resolve_r10m_vegetation_bands(safe_path)
            outputs.append(
                process_sentinel2_scene(
                    bands=bands,
                    output_root=config.output_dir,
                    reflectance_scale=config.reflectance_scale,
                    max_preview_size=config.max_preview_size,
                    overwrite=config.overwrite,
                    flat_output=flat_output,
                    methane_proxy=config.methane_proxy,
                )
            )
        except Exception as exc:
            failures.append(f"{safe_path}: {exc}")

    if failures:
        details = "\n".join(failures)
        raise RuntimeError(f"{len(failures)} Sentinel-2 scene(s) failed:\n{details}")
    return outputs


def run_final_sentinel2_batch(config: VegetationPipelineConfig) -> list[SceneOutputs]:
    """Run the final publication-ready Sentinel-2 vegetation proxy pipeline."""

    return run_vegetation_batch(config)
