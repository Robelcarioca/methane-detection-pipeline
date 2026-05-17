"""Final CLI entry point for Sentinel-2 vegetation methane proxy processing."""

from __future__ import annotations

import argparse
from pathlib import Path

from anomaly_detection.methane_proxy import MethaneProxyConfig
from preprocessing.vegetation_pipeline import VegetationPipelineConfig, run_final_sentinel2_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate NDVI, vegetation stress, NDVI anomaly, methane risk, and cluster products."
    )
    parser.add_argument("--input", default="raw_data", help="Directory containing one or more Sentinel-2 .SAFE products.")
    parser.add_argument("--output", default="outputs/final", help="Directory where final products are written.")
    parser.add_argument("--max-preview-size", type=int, default=2048, help="Largest PNG preview dimension in pixels.")
    parser.add_argument("--reflectance-scale", type=float, default=10000.0, help="Sentinel-2 reflectance scale factor.")
    parser.add_argument("--smoothing-sigma", type=float, default=3.0, help="Gaussian sigma for local NDVI smoothing.")
    parser.add_argument("--medium-anomaly-z", type=float, default=1.0, help="NDVI deficit z-score for medium risk.")
    parser.add_argument("--high-anomaly-z", type=float, default=1.75, help="NDVI deficit z-score for high risk.")
    parser.add_argument("--min-cluster-pixels", type=int, default=9, help="Minimum connected pixels retained as a cluster.")
    parser.add_argument("--no-overwrite", action="store_true", help="Reuse existing final raster/array outputs when present.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    methane_proxy = MethaneProxyConfig(
        smoothing_sigma=args.smoothing_sigma,
        medium_anomaly_z=args.medium_anomaly_z,
        high_anomaly_z=args.high_anomaly_z,
        min_cluster_pixels=args.min_cluster_pixels,
    )
    config = VegetationPipelineConfig(
        raw_data=Path(args.input),
        output_dir=Path(args.output),
        reflectance_scale=args.reflectance_scale,
        max_preview_size=args.max_preview_size,
        overwrite=not args.no_overwrite,
        flat_output=True,
        methane_proxy=methane_proxy,
    )
    outputs = run_final_sentinel2_batch(config)

    print(f"Processed {len(outputs)} Sentinel-2 scene(s).")
    for scene in outputs:
        print(f"- {scene.scene_id}: {scene.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
