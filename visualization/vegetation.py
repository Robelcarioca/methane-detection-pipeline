"""Downsampled visualization writers for vegetation products."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_rgb_png(rgb: np.ndarray, output_path: Path) -> Path:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
    import warnings

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "PNG",
        "height": rgb.shape[0],
        "width": rgb.shape[1],
        "count": 3,
        "dtype": "uint8",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(target, "w", **profile) as dst:
            dst.write(np.moveaxis(rgb, -1, 0))
    return target


def _linear_colormap(values: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    valid = np.isfinite(values)
    clipped = np.clip(values, stops[0][0], stops[-1][0])
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)

    for (left_value, left_color), (right_value, right_color) in zip(stops[:-1], stops[1:]):
        mask = valid & (clipped >= left_value) & (clipped <= right_value)
        if not np.any(mask):
            continue
        span = max(right_value - left_value, 1e-6)
        weight = ((clipped[mask] - left_value) / span).astype(np.float32)
        left = np.asarray(left_color, dtype=np.float32)
        right = np.asarray(right_color, dtype=np.float32)
        rgb[mask] = np.round(left + (right - left) * weight[:, np.newaxis]).astype(np.uint8)
    return rgb


def save_ndvi_png(ndvi: np.ndarray, output_path: str | Path) -> Path:
    """Save a compact NDVI visualization without axes or full-res rendering."""

    stops = [
        (-1.0, (165, 0, 38)),
        (0.0, (255, 255, 191)),
        (1.0, (0, 104, 55)),
    ]
    rgb = _linear_colormap(ndvi.astype(np.float32, copy=False), stops)
    return _write_rgb_png(rgb, Path(output_path))


def save_stress_png(stress_map: np.ndarray, output_path: str | Path, nodata_value: int = 255) -> Path:
    """Save a compact categorical vegetation stress map."""

    palette = np.array(
        [
            [215, 48, 39],
            [254, 224, 139],
            [26, 152, 80],
        ],
        dtype=np.uint8,
    )
    rgb = np.zeros((*stress_map.shape, 3), dtype=np.uint8)
    valid = (stress_map != nodata_value) & (stress_map <= 2)
    rgb[valid] = palette[stress_map[valid]]
    return _write_rgb_png(rgb, Path(output_path))


def save_methane_risk_png(risk_map: np.ndarray, output_path: str | Path) -> Path:
    """Save a compact categorical methane proxy risk map."""

    palette = np.array(
        [
            [35, 139, 69],
            [254, 178, 76],
            [215, 25, 28],
        ],
        dtype=np.uint8,
    )
    rgb = np.zeros((*risk_map.shape, 3), dtype=np.uint8)
    valid = risk_map <= 2
    rgb[valid] = palette[risk_map[valid]]
    return _write_rgb_png(rgb, Path(output_path))
