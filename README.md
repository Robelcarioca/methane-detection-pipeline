# MethaneSAT 2026 Sentinel-2 Methane Plume Pipeline

Production-oriented research skeleton for methane plume detection from Sentinel-2 multispectral imagery. The system is configuration-driven, modular, and designed for scaling from notebook experiments to batch pipelines.

## Architecture

- `data_ingestion/`: Sentinel-2 query, download, metadata, retries, and file organization.
- `preprocessing/`: cloud masking, reprojection, resampling, normalization, temporal pairing, and patch extraction.
- `temporal_alignment/`: temporal matching, co-registration, CRS alignment, and weather/wind synchronization hooks.
- `feature_engineering/`: SWIR ratios, temporal differences, anomaly features, texture, and masks.
- `dataset_builder/`: leakage-aware splits, tensor export, HDF5 generation, and PyTorch datasets.
- `models/`: CNN, U-Net, ViT encoder starter modules, training, validation, and inference.
- `explainability/`: SHAP, Grad-CAM, saliency, and attention map utilities.
- `visualization/`: RGB, plume overlay, confidence maps, spectra, and time series.
- `utils/`: configuration, logging, geospatial IO, exceptions, and reproducibility helpers.

## Quick Start

Use Python 3.11 or 3.12 for the full geospatial/ML environment. Python 3.14 may not have wheels for every package in the scientific stack yet.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

python scripts/ingest_sentinel2.py --config configs/config.yaml
python scripts/preprocess_dataset.py --config configs/config.yaml
python scripts/train_model.py --config configs/config.yaml
python scripts/visualize_outputs.py --config configs/config.yaml
```

## Final Sentinel-2 Methane Proxy Pipeline

For local `.SAFE` products already downloaded into `raw_data`, run the final
windowed processing pipeline:

```bash
python process_sentinel2.py --input raw_data --output outputs/final
```

For a single SAFE product, final products are written directly into
`outputs/final/`:

- `ndvi.tif` and `stress_map.tif` GeoTIFFs using the Sentinel-2 10 m grid.
- `methane_risk_map.tif` with classes `0` low, `1` medium, and `2` high risk.
- `anomaly_map.npy` containing the smoothed NDVI-deficit z-score anomaly.
- `cluster_map.npy` containing connected stress/risk zone labels.
- `ndvi.png` and `methane_risk.png` downsampled previews for stable visualization.

Stress classes are `0` for stressed/non-vegetated, `1` for moderate vegetation,
`2` for healthy vegetation, and `255` for nodata.

The methane proxy is intentionally interpretable: it computes NDVI in raster
windows, estimates scene-level NDVI statistics, smooths local NDVI with a
Gaussian filter, converts local deficits into z-score anomalies, assigns
vegetation-supported methane risk classes, and labels contiguous risk clusters
without loading full-resolution float rasters into RAM.

The older vegetation-only command remains available for compatibility:

```bash
python scripts/process_sentinel2_vegetation.py --raw-data raw_data --output-dir outputs/vegetation
```

For a faster skeleton verification on Windows:

```powershell
.\scripts\bootstrap_env.ps1
```

To attempt the complete scientific dependency stack:

```powershell
.\scripts\bootstrap_env.ps1 -Full
```

If pip reports no matching packages or silently fails to resolve packages, check whether `PIP_NO_INDEX` is set:

```powershell
Remove-Item Env:PIP_NO_INDEX -ErrorAction SilentlyContinue
python scripts/check_environment.py
```

Credentials for Sentinel Hub or Google Earth Engine should be provided through environment variables or secure local config overrides. Do not commit secrets.

## Data Layout

Raw Sentinel-2 downloads are organized as:

```text
raw_data/{region_id}/{YYYY-MM-DD}/{tile_id}/
```

Processed patches and tensors are written to:

```text
processed_data/
datasets/
outputs/
logs/
```

## Scientific Defaults

- Sentinel-2 Level-1C SWIR bands `B11` and `B12` are first-class channels.
- All bands are aligned to 20 m by default.
- Patch tensors use channel-first layout: `[channels, height, width]`.
- Default patch shape is `[20, 128, 128]`.
- Temporal examples are paired as `(t-1, t)` for multi-temporal plume detection.

## Current State

This is a complete architecture and starter skeleton. Provider-specific API calls are isolated behind reusable interfaces so deeper production behavior can be implemented without changing downstream pipeline contracts.
