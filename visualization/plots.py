"""Visualization utilities for methane detection outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rgb_composite(stack: np.ndarray, band_indices: tuple[int, int, int] = (2, 1, 0)) -> np.ndarray:
    """Create an RGB image from a channel-first stack."""

    rgb = stack[list(band_indices)]
    rgb = np.moveaxis(rgb, 0, -1)
    return np.clip(rgb, 0.0, 1.0)


def save_plume_overlay(
    rgb: np.ndarray,
    confidence: np.ndarray,
    output_path: str | Path,
    threshold: float = 0.5,
    colormap: str = "inferno",
) -> Path:
    """Save methane plume confidence overlay."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    masked = np.ma.masked_less(confidence, threshold)
    plt.imshow(masked, cmap=colormap, alpha=0.55)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(target, dpi=180)
    plt.close()
    return target


def save_spectral_profile(values: np.ndarray, output_path: str | Path) -> Path:
    """Save a spectral profile plot."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(values)
    plt.xlabel("Channel")
    plt.ylabel("Reflectance / Feature Value")
    plt.tight_layout()
    plt.savefig(target, dpi=160)
    plt.close()
    return target
