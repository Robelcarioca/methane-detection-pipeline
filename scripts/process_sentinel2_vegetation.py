"""Compatibility CLI for batch Sentinel-2 vegetation proxy products."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.vegetation_pipeline import VegetationPipelineConfig, run_vegetation_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NDVI, vegetation stress, anomaly, methane risk, and cluster maps from Sentinel-2 SAFE scenes."
    )
    parser.add_argument("--raw-data", default="raw_data", help="Directory containing one or more Sentinel-2 .SAFE products.")
    parser.add_argument("--output-dir", default="outputs/vegetation", help="Directory where per-scene outputs are written.")
    parser.add_argument("--max-preview-size", type=int, default=2048, help="Largest PNG preview dimension in pixels.")
    parser.add_argument("--reflectance-scale", type=float, default=10000.0, help="Sentinel-2 integer reflectance scale factor.")
    parser.add_argument("--no-overwrite", action="store_true", help="Reuse existing GeoTIFF/NumPy outputs when present.")
    args = parser.parse_args()

    config = VegetationPipelineConfig(
        raw_data=Path(args.raw_data),
        output_dir=Path(args.output_dir),
        reflectance_scale=args.reflectance_scale,
        max_preview_size=args.max_preview_size,
        overwrite=not args.no_overwrite,
    )
    outputs = run_vegetation_batch(config)

    print(f"Processed {len(outputs)} Sentinel-2 scene(s).")
    for scene in outputs:
        print(f"- {scene.scene_id}: {scene.output_dir}")


if __name__ == "__main__":
    main()
