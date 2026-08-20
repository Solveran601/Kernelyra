from __future__ import annotations

from typing import Any

import numpy as np


def _finite(metrics: dict[str, Any]) -> dict[str, Any]:
    for key, value in metrics.items():
        if isinstance(value, int | float) and not np.isfinite(float(value)):
            raise FloatingPointError(f"Metric {key} became NaN/Inf")
    return metrics


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, loss: float) -> dict[str, Any]:
    truth = y_true.astype(np.int64)
    predicted = (probabilities >= .5).astype(np.int64)
    tp = int(((predicted == 1) & (truth == 1)).sum())
    fp = int(((predicted == 1) & (truth == 0)).sum())
    fn = int(((predicted == 0) & (truth == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    positives = probabilities[truth == 1]
    negatives = probabilities[truth == 0]
    auc: float | None = None
    if len(positives) and len(negatives):
        comparisons = (positives[:, None] > negatives[None, :]).mean()
        ties = (positives[:, None] == negatives[None, :]).mean()
        auc = float(comparisons + .5 * ties)
    return _finite(
        {
            "loss": float(loss),
            "accuracy": float((predicted == truth).mean()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": auc,
        }
    )


def multiclass_metrics(y_true: np.ndarray, probabilities: np.ndarray, loss: float) -> dict[str, Any]:
    truth = y_true.astype(np.int64)
    predicted = probabilities.argmax(axis=1)
    classes = sorted(set(truth.tolist()) | set(predicted.tolist()))
    f1_values: list[float] = []
    confusion: list[list[int]] = []
    for actual in classes:
        row: list[int] = []
        for guess in classes:
            row.append(int(((truth == actual) & (predicted == guess)).sum()))
        confusion.append(row)
        tp = int(((truth == actual) & (predicted == actual)).sum())
        fp = int(((truth != actual) & (predicted == actual)).sum())
        fn = int(((truth == actual) & (predicted != actual)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1_values.append(2 * precision * recall / max(1e-12, precision + recall))
    accuracy = float((predicted == truth).mean())
    return _finite(
        {
            "loss": float(loss),
            "accuracy": accuracy,
            "macro_f1": float(np.mean(f1_values)),
            "micro_f1": accuracy,
            "confusion_matrix": confusion,
            "classes": classes,
        }
    )


def regression_metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    truth = y_true.astype(np.float64)
    output = predicted.astype(np.float64)
    error = output - truth
    mse = float(np.mean(error**2))
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / denominator if denominator > 1e-12 else 0.0
    return _finite(
        {
            "loss": mse,
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2),
        }
    )
