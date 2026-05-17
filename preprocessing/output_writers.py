"""Output writers for processed satellite tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def write_numpy(array: np.ndarray, output_path: str | Path) -> Path:
    """Write a tensor as NumPy `.npy`."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, array)
    return target


def write_hdf5(array: np.ndarray, output_path: str | Path, dataset_name: str = "patch") -> Path:
    """Write a tensor to HDF5."""

    import h5py

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as handle:
        handle.create_dataset(dataset_name, data=array.astype(np.float32), compression="gzip")
    return target


def write_zarr(array: np.ndarray, output_path: str | Path, dataset_name: str = "patch") -> Path:
    """Write a tensor to Zarr."""

    import zarr

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(target), mode="w")
    group.create_dataset(dataset_name, data=array.astype(np.float32), chunks=(1, *array.shape[-2:]))
    return target


def write_geotiff(
    array: np.ndarray,
    output_path: str | Path,
    profile: dict[str, Any],
) -> Path:
    """Write a 2D raster or channel-first tensor as GeoTIFF."""

    import rasterio

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raster = array[np.newaxis, ...] if array.ndim == 2 else array
    if raster.ndim != 3:
        raise ValueError("GeoTIFF arrays must be 2D or channel-first 3D.")

    write_profile = profile.copy()
    write_profile.update(count=raster.shape[0], dtype=str(raster.dtype), compress=write_profile.get("compress", "deflate"))
    with rasterio.open(target, "w", **write_profile) as dataset:
        dataset.write(raster)
    return target
