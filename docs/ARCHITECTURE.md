# Architecture

This repository is organized as a set of independent Python packages that can be composed into larger research workflows.

## Pipeline Flow

1. `data_ingestion` queries Sentinel-2 Level-1C scenes by coordinates, polygon, and date range.
2. `preprocessing` aligns bands, masks clouds, normalizes reflectance, builds temporal pairs, and extracts channel-first patches.
3. `temporal_alignment` matches `(t-1, t)` observations and provides co-registration and weather synchronization hooks.
4. `feature_engineering` computes SWIR methane-sensitive ratios, delta reflectance, anomalies, texture placeholders, and masks.
5. `dataset_builder` prevents geographic leakage, tracks metadata, and exports ML-ready tensors.
6. `models` trains, validates, and runs inference for CNN, U-Net, and transformer/U-Net experiments.
7. `explainability` produces SHAP, attention, Grad-CAM, and saliency outputs.
8. `visualization` creates RGB composites, plume overlays, spectral views, time-series plots, and confidence maps.

## Configuration

All major paths, bands, query constraints, preprocessing parameters, model settings, and training settings live in `configs/config.yaml`.

Provider credentials should be supplied through environment variables or private local override files. Secrets should not be committed.

## Scientific Contracts

- Imagery is treated as channel-first arrays: `[channels, height, width]`.
- Default patch tensors are `[20, 128, 128]`.
- SWIR bands `B11` and `B12` are required because they are central to methane-sensitive spectral features.
- Multi-temporal model inputs use paired observations `(t-1, t)`.
- Geographic groups should be held out together to reduce leakage across train, validation, and test splits.

## Scaling Path

The current skeleton keeps provider calls and heavy geospatial operations behind small interfaces. The next production step is to replace placeholders with:

- Sentinel Hub Process API or Google Earth Engine export calls.
- Rasterio/rioxarray reprojection and resampling against reference grids.
- Dask/xarray/Zarr chunked processing for large regions.
- Experiment tracking and distributed training.
