"""HDF5 dataset generation."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def write_hdf5_dataset(
    tensors: list[np.ndarray],
    labels: list[np.ndarray],
    output_path: str | Path,
    tensor_key: str = "x",
    label_key: str = "y",
) -> Path:
    """Write ML-ready tensors and labels to HDF5."""

    if len(tensors) != len(labels):
        raise ValueError("Tensors and labels must have the same length.")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as handle:
        handle.create_dataset(tensor_key, data=np.stack(tensors), compression="gzip")
        handle.create_dataset(label_key, data=np.stack(labels), compression="gzip")
    return target
