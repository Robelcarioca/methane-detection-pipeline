"""Patch extraction for channel-first satellite tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class PatchWindow:
    """Pixel coordinates for an extracted patch."""

    row: int
    col: int
    height: int
    width: int


def extract_patches(
    array: np.ndarray,
    patch_size: int = 128,
    stride: int | None = None,
    drop_incomplete: bool = True,
) -> Iterator[tuple[np.ndarray, PatchWindow]]:
    """Yield channel-first patches shaped `[channels, patch_size, patch_size]`."""

    if array.ndim != 3:
        raise ValueError("Expected array shape [channels, height, width].")

    stride = stride or patch_size
    _, height, width = array.shape

    for row in range(0, height, stride):
        for col in range(0, width, stride):
            patch = array[:, row : row + patch_size, col : col + patch_size]
            if drop_incomplete and patch.shape[-2:] != (patch_size, patch_size):
                continue
            yield patch.astype(np.float32), PatchWindow(row=row, col=col, height=patch.shape[-2], width=patch.shape[-1])
