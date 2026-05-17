"""Train/validation/test split utilities with geographic leakage prevention."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def geographic_split(
    metadata: pd.DataFrame,
    group_column: str = "region_id",
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by geographic group to prevent leakage."""

    if group_column not in metadata:
        raise ValueError(f"Missing group column: {group_column}")

    splitter = GroupShuffleSplit(n_splits=1, train_size=train_fraction, random_state=seed)
    train_idx, holdout_idx = next(splitter.split(metadata, groups=metadata[group_column]))
    train = metadata.iloc[train_idx].copy()
    holdout = metadata.iloc[holdout_idx].copy()

    relative_val = val_fraction / max(1e-6, 1.0 - train_fraction)
    val, test = train_test_split(
        holdout,
        train_size=relative_val,
        random_state=seed,
        stratify=None,
    )
    return train, val.copy(), test.copy()
