"""Small, serialisable safety checks shared by every training backend."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any


def _non_finite_paths(value: Any, path: str = "metrics") -> list[str]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, item in value.items():
            result.extend(_non_finite_paths(item, f"{path}.{key}"))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            result.extend(_non_finite_paths(item, f"{path}[{index}]"))
        return result
    if isinstance(value, Real) and not isinstance(value, bool) and not math.isfinite(float(value)):
        return [path]
    return []


class QualityGate:
    """Validate a training observation before it can replace a safe checkpoint."""

    def __init__(self, *, degradation_margin: float, baseline_score: float | None = None):
        self.degradation_margin = float(degradation_margin)
        self.baseline_score = float(baseline_score) if baseline_score is not None else None

    def inspect(
        self,
        *,
        score: float,
        loss: float,
        metrics: Mapping[str, Any],
        best_score: float,
    ) -> dict[str, Any]:
        score_value = float(score)
        loss_value = float(loss)
        bad_paths = _non_finite_paths(
            {"score": score_value, "loss": loss_value, "validation": metrics}, "quality"
        )
        finite = not bad_paths
        baseline_gap = (
            max(0.0, self.baseline_score - score_value)
            if finite and self.baseline_score is not None
            else None
        )
        best_gap = max(0.0, float(best_score) - score_value) if finite else None
        return {
            "status": "invalid" if bad_paths else "regressing" if best_gap > self.degradation_margin else "stable",
            "finite": finite,
            "invalid_paths": bad_paths,
            "score": score_value if math.isfinite(score_value) else None,
            "loss": loss_value if math.isfinite(loss_value) else None,
            "best_gap": best_gap,
            "baseline_score": self.baseline_score,
            "baseline_gap": baseline_gap,
            "degradation_margin": self.degradation_margin,
        }
