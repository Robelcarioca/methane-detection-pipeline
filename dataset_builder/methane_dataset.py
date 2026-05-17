"""PyTorch datasets for methane plume detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore[assignment]


class MethanePatchDataset(Dataset):
    """HDF5-backed PyTorch dataset for patch tensors and labels."""

    def __init__(
        self,
        hdf5_path: str | Path,
        tensor_key: str = "x",
        label_key: str = "y",
        transform: Any | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to use MethanePatchDataset.")
        self.hdf5_path = Path(hdf5_path)
        self.tensor_key = tensor_key
        self.label_key = label_key
        self.transform = transform
        with h5py.File(self.hdf5_path, "r") as handle:
            self.length = int(handle[self.tensor_key].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        with h5py.File(self.hdf5_path, "r") as handle:
            x = np.asarray(handle[self.tensor_key][index], dtype=np.float32)
            y = np.asarray(handle[self.label_key][index], dtype=np.float32)

        sample = {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
