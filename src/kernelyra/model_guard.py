"""Trend-aware extension of the finite-value Quality Gate."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .quality import QualityGate


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


class ModelGuard:
    """Side-effect-free public Model Guard evaluator.

    The runtime uses the same finite-value and trend rules. This facade lets
    an application inspect its own validation loop before deciding whether to
    persist a checkpoint or stop. It never creates, restores, or deletes a
    model by itself.
    """

    def __init__(
        self,
        *,
        degradation_margin: float = .03,
        degradation_patience: int = 3,
        baseline_score: float | None = None,
    ):
        margin = float(degradation_margin)
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("degradation_margin must be a finite number greater than or equal to zero")
        patience = int(degradation_patience)
        if patience < 1:
            raise ValueError("degradation_patience must be at least one")
        if baseline_score is not None and not math.isfinite(float(baseline_score)):
            raise ValueError("baseline_score must be finite when provided")
        self.degradation_margin = margin
        self.degradation_patience = patience
        self.baseline_score = float(baseline_score) if baseline_score is not None else None

    def inspect(
        self,
        *,
        score: float,
        loss: float,
        metrics: dict[str, Any],
        best_score: float,
        scores: Sequence[float] = (),
    ) -> dict[str, Any]:
        """Combine finite-metric and bounded score-trend evidence.

        ``scores`` is caller-owned history. The method returns evidence only;
        callers choose their checkpoint policy and any response to an alert.
        """
        quality = QualityGate(
            degradation_margin=self.degradation_margin,
            baseline_score=self.baseline_score,
        ).inspect(score=score, loss=loss, metrics=metrics, best_score=best_score)
        trend = assess_trend(
            scores,
            best_score=best_score,
            degradation_margin=self.degradation_margin,
            degradation_patience=self.degradation_patience,
            baseline_score=self.baseline_score,
        )
        status = "invalid" if quality["status"] == "invalid" else trend["status"]
        return {
            "status": status,
            "restore_recommended": bool(trend["restore_recommended"]),
            "quality": quality,
            "trend": trend,
            "contract": "inspection only; this object never changes checkpoints or model parameters",
        }
