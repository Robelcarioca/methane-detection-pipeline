import numpy as np

from feature_engineering.features import temporal_difference


def test_temporal_difference() -> None:
    previous = np.zeros((2, 4, 4), dtype=np.float32)
    current = np.ones((2, 4, 4), dtype=np.float32)
    diff = temporal_difference(previous, current)
    assert diff.mean() == 1.0
