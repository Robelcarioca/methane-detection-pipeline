"""Example visualization workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from visualization.plots import rgb_composite, save_plume_overlay


def run_visualization(config: dict[str, Any]) -> Path:
    """Create a placeholder plume overlay from random confidence data."""

    output_dir = Path(config["paths"]["outputs"]) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    stack = np.zeros((3, 128, 128), dtype=np.float32)
    confidence = np.zeros((128, 128), dtype=np.float32)
    return save_plume_overlay(
        rgb=rgb_composite(stack),
        confidence=confidence,
        output_path=output_dir / "example_plume_overlay.png",
        threshold=float(config["visualization"].get("confidence_threshold", 0.5)),
        colormap=config["visualization"].get("plume_colormap", "inferno"),
    )
