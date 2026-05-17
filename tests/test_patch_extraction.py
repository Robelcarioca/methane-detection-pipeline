import numpy as np

from preprocessing.patch_extraction import extract_patches


def test_extract_patches_channel_first() -> None:
    array = np.zeros((20, 256, 256), dtype=np.float32)
    patches = list(extract_patches(array, patch_size=128))
    assert len(patches) == 4
    assert patches[0][0].shape == (20, 128, 128)
