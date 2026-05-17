import numpy as np
import pytest
from pathlib import Path
from uuid import uuid4

from preprocessing.output_writers import write_hdf5, write_numpy


def _artifact_dir() -> Path:
    path = Path("manual_tmp") / "test_artifacts" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_numpy_and_hdf5_writers() -> None:
    output_dir = _artifact_dir()
    array = np.zeros((2, 4, 4), dtype=np.float32)
    assert write_numpy(array, output_dir / "patch.npy").exists()

    pytest.importorskip("h5py")
    assert write_hdf5(array, output_dir / "patch.h5").exists()
