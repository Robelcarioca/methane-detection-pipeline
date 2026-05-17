"""Example preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from preprocessing.band_processing import align_and_resample_bands, fill_missing, normalize_reflectance
from preprocessing.patch_extraction import extract_patches
from utils.io import ensure_dir, save_numpy


def preprocess_scene(
    band_paths: dict[str, Path],
    output_dir: str | Path,
    patch_size: int = 128,
    target_resolution_m: int = 20,
) -> list[Path]:
    """Align bands, normalize reflectance, and write NumPy patch tensors."""

    output = ensure_dir(output_dir)
    stack = align_and_resample_bands(band_paths, target_resolution_m=target_resolution_m)
    stack = fill_missing(normalize_reflectance(stack))

    written: list[Path] = []
    for index, (patch, window) in enumerate(extract_patches(stack, patch_size=patch_size)):
        patch_path = output / f"patch_{index:05d}_r{window.row}_c{window.col}.npy"
        save_numpy(patch_path, patch)
        written.append(patch_path)
    return written


def run_preprocessing(config: dict[str, Any]) -> int:
    """Run a minimal preprocessing pass over discovered raw scene folders."""

    raw_root = Path(config["paths"]["raw_data"])
    processed_root = Path(config["paths"]["processed_data"])
    patch_size = int(config["preprocessing"]["patch_size"])
    target_resolution_m = int(config["preprocessing"]["target_resolution_m"])
    bands = config["ingestion"]["bands"]["required"] + config["ingestion"]["bands"].get("optional_rgb", [])

    count = 0
    for scene_dir in raw_root.glob("*/*/*"):
        band_paths = {band: scene_dir / f"{band}.tif" for band in bands}
        output_dir = processed_root / scene_dir.relative_to(raw_root)
        count += len(preprocess_scene(band_paths, output_dir, patch_size, target_resolution_m))
    return count
