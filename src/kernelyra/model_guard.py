"""Trend-aware extension of the finite-value Quality Gate."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def assess_trend(
    scores: Sequence[float],
    *,
    best_score: float,
    degradation_margin: float,
    degradation_patience: int,
    baseline_score: float | None = None,
) -> dict[str, Any]:
    """Return bounded evidence for Model Guard V2 without changing checkpoints itself."""
    window = [float(item) for item in scores[-8:] if math.isfinite(float(item))]
    if not window:
        return {"status": "insufficient_evidence", "window": 0, "restore_recommended": False}
    slope = 0.0 if len(window) < 2 else (window[-1] - window[0]) / (len(window) - 1)
    threshold = float(best_score) - float(degradation_margin)
    consecutive_below_best = 0
    for score in reversed(window):
        if score < threshold:
            consecutive_below_best += 1
        else:
            break
    baseline_gap = (
        max(0.0, float(baseline_score) - window[-1]) if baseline_score is not None else None
    )
    restore = consecutive_below_best >= int(degradation_patience)
    return {
        "status": "restore_recommended" if restore else "regressing" if window[-1] < threshold else "stable",
        "window": len(window),
        "scores": window,
        "trend_per_evaluation": slope,
        "best_score": float(best_score),
        "baseline_score": float(baseline_score) if baseline_score is not None else None,
        "baseline_gap": baseline_gap,
        "degradation_margin": float(degradation_margin),
        "consecutive_below_best": consecutive_below_best,
        "restore_recommended": restore,
    }
