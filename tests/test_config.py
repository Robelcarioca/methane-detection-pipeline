from pathlib import Path

from utils.config import load_config


def test_load_config() -> None:
    config = load_config(Path("configs/config.yaml"))
    assert config.section("project")["name"] == "methanesat_2026_sentinel2_pipeline"
