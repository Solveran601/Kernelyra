"""Bounded, JSON-ready training traces for terminal diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

TRACE_LIMIT = 128


@dataclass(slots=True)
class TrainingTrace:
    """Keep a compact event history without allowing metrics to grow forever."""

    events: list[dict[str, Any]] = field(default_factory=list)
    limit: int = TRACE_LIMIT

    @classmethod
    def from_metrics(cls, metrics: dict[str, Any]) -> TrainingTrace:
        raw = metrics.get("trace")
        events = [dict(item) for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
        return cls(events[-TRACE_LIMIT:])

    def add(self, event: str, **values: Any) -> None:
        self.events.append({"event": event, "at": time.time(), **values})
        if len(self.events) > self.limit:
            del self.events[: len(self.events) - self.limit]

    def summary(self) -> dict[str, Any]:
        evaluations = [item for item in self.events if item.get("event") == "evaluation"]
        latest = evaluations[-1] if evaluations else None
        return {
            "events": len(self.events),
            "evaluations": len(evaluations),
            "latest_step": latest.get("step") if latest else None,
            "latest_score": latest.get("score") if latest else None,
            "latest_updates_per_second": latest.get("updates_per_second") if latest else None,
        }
