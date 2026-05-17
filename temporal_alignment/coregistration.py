"""Spatial co-registration and CRS alignment hooks."""

from __future__ import annotations

from pathlib import Path


def coregister_to_reference(source_path: Path, reference_path: Path, output_path: Path) -> Path:
    """Co-register source raster to a reference raster grid.

    The starter skeleton preserves the function boundary. Production code should
    call rasterio/rioxarray reprojection and, if needed, image-registration
    refinement.
    """

    _ = reference_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.exists() and source_path != output_path:
        output_path.write_bytes(source_path.read_bytes())
    else:
        output_path.touch()
    return output_path
