"""Deterministic materialized train/validation/test partitions."""

from __future__ import annotations

import numpy as np

from ..models import TaskType


def _sizes(total: int, validation_fraction: float, test_fraction: float) -> tuple[int, int, int]:
    validation = max(16, int(round(total * validation_fraction)))
    test = max(1, int(round(total * test_fraction)))
    train = total - validation - test
    if train < 8:
        raise ValueError("Not enough rows after deterministic train/validation/test split")
    return train, validation, test


def _stratified_indices(y: np.ndarray, *, seed: int, validation_fraction: float, test_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in np.unique(y):
        values = rng.permutation(np.flatnonzero(y == label))
        test_count = int(round(len(values) * test_fraction))
        validation_count = int(round(len(values) * validation_fraction))
        if len(values) >= 3:
            test_count = max(1, test_count)
            validation_count = max(1, validation_count)
        test_count = min(test_count, max(0, len(values) - 1))
        validation_count = min(validation_count, max(0, len(values) - test_count - 1))
        test_parts.append(values[:test_count])
        validation_parts.append(values[test_count : test_count + validation_count])
        train_parts.append(values[test_count + validation_count :])
    train = np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int64)
    validation = np.concatenate(validation_parts) if validation_parts else np.empty(0, dtype=np.int64)
    test = np.concatenate(test_parts) if test_parts else np.empty(0, dtype=np.int64)
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    if len(train) < 8 or len(validation) < 1 or len(test) < 1:
        raise ValueError("Not enough class coverage after stratified train/validation/test split")
    return train, validation, test


def split_arrays(
    x: np.ndarray,
    y: np.ndarray,
    *,
    task_type: str,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    strategy: str = "random",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition arrays with an explicit policy used in the stored run contract."""
    train_size, validation_size, test_size = _sizes(len(x), validation_fraction, test_fraction)
    if strategy == "stratified" and task_type in {
        TaskType.BINARY_CLASSIFICATION.value,
        TaskType.MULTICLASS_CLASSIFICATION.value,
    }:
        train_index, validation_index, test_index = _stratified_indices(
            y, seed=seed, validation_fraction=validation_fraction, test_fraction=test_fraction
        )
        return (
            x[train_index], y[train_index], x[validation_index], y[validation_index], x[test_index], y[test_index]
        )
    if strategy == "temporal":
        order = np.arange(len(x))
    else:
        order = np.random.default_rng(seed).permutation(len(x))
    x, y = x[order], y[order]
    train_end = train_size
    validation_end = train_end + validation_size
    return x[:train_end], y[:train_end], x[train_end:validation_end], y[train_end:validation_end], x[validation_end:validation_end + test_size], y[validation_end:validation_end + test_size]
