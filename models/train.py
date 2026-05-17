"""Training and validation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dataset_builder.methane_dataset import MethanePatchDataset
from models.cnn_baseline import CNNBaseline
from models.losses import DiceBCELoss
from models.unet import UNet
from models.vit_unet import ViTUNet


def resolve_device(device_config: str = "auto") -> torch.device:
    """Resolve CPU/GPU device from config."""

    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    """Instantiate a model from config."""

    model_config = config["model"]
    architecture = model_config.get("architecture", "unet")
    kwargs = {
        "in_channels": int(model_config.get("in_channels", 20)),
        "out_channels": int(model_config.get("out_channels", 1)),
        "base_channels": int(model_config.get("base_channels", 32)),
    }
    if architecture == "cnn_baseline":
        return CNNBaseline(**kwargs)
    if architecture == "vit_unet":
        return ViTUNet(
            **kwargs,
            encoder_name=model_config.get("encoder_name", "vit_base_patch16_224"),
            pretrained=bool(model_config.get("pretrained", False)),
        )
    return UNet(**kwargs)


def train(config: dict[str, Any]) -> Path:
    """Run an example training loop and save the latest checkpoint."""

    device = resolve_device(config["project"].get("device", "auto"))
    dataset_config = config["dataset"]
    training_config = config["training"]

    dataset = MethanePatchDataset(
        dataset_config["hdf5_path"],
        tensor_key=dataset_config.get("tensor_key", "x"),
        label_key=dataset_config.get("label_key", "y"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_config.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
    )

    model = build_model(config).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_config.get("learning_rate", 1e-4)))

    model.train()
    for _epoch in range(int(training_config.get("epochs", 1))):
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    checkpoint_dir = Path(training_config.get("checkpoint_dir", "outputs/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "latest.pt"
    torch.save({"model_state": model.state_dict(), "config": config}, checkpoint_path)
    return checkpoint_path
