"""Tensor export helpers for NumPy, HDF5, and Zarr."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_numpy_tensor(tensor: np.ndarray, output_path: str | Path) -> Path:
    """Export a single tensor as `.npy`."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, tensor.astype(np.float32))
    return target
