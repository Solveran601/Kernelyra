from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(slots=True)
class BackendConfig:
    x: np.ndarray | None
    y: np.ndarray | None
    profile: str
    seed: int
    task_type: str = "binary_classification"
    validation_fraction: float = .15
    test_fraction: float = .15
    resource_limits: dict[str, Any] = field(default_factory=dict)
    model_path: Path | None = None
    checkpoint_path: Path | None = None
    dataset_spec: dict[str, Any] | None = None
    learning_rate: float | None = None
    weight_decay: float = 0.0
    hidden_layers: tuple[int, ...] = ()
    precision: str = "auto"
    data_workers: int = 0
    prefetch: int = 1


@dataclass(slots=True)
class TrainingSession:
    state: Any
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    rng: np.random.Generator
    metadata: dict[str, Any] = field(default_factory=dict)
    data_source: Any | None = None


@dataclass(frozen=True, slots=True)
class StepResult:
    loss: float
    samples: int


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    score: float
    metrics: dict[str, Any] = field(default_factory=dict)


class TrainingBackend(Protocol):
    name: str

    def inspect_model(self, path: Path) -> dict[str, Any]: ...
    def create_session(self, config: BackendConfig) -> TrainingSession: ...
    def train_step(self, session: TrainingSession, batch_size: int) -> StepResult: ...
    def evaluate(self, session: TrainingSession) -> EvaluationResult: ...
    def evaluate_test(self, session: TrainingSession) -> EvaluationResult: ...
    def save_checkpoint(self, session: TrainingSession, path: Path, metadata: dict[str, Any]) -> None: ...
    def restore_checkpoint(self, session: TrainingSession, path: Path) -> None: ...
