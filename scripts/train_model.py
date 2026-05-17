"""CLI: train methane detection model."""

from __future__ import annotations

import argparse

from models.train import train
from utils.config import load_config
from utils.logging import configure_logging
from utils.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train methane plume segmentation model.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = configure_logging(config.section("paths").get("logs", "logs"))
    seed_everything(int(config.section("project").get("seed", 42)))
    checkpoint = train(config.data)
    logger.info("Training complete: %s", checkpoint)


if __name__ == "__main__":
    main()
