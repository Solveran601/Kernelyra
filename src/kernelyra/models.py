from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class RunStatus(str, Enum):
    DRAFT = "draft"
    QUEUED = "queued"
    TRAINING = "training"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR_RECOVERABLE = "error_recoverable"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    columns: tuple[dict[str, Any], ...]
    target: str
    target_dtype: str
    task_types: tuple[str, ...]
    feature_count: int
    row_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    source_name: str
    source_kind: str
    format: str
    sha256: str
    size_bytes: int
    row_count: int
    schema: dict[str, Any]
    task_compatibility: tuple[str, ...]
    transformations: tuple[dict[str, Any], ...]
    split_seed: int
    warnings: tuple[str, ...]
    created_at: float
    ingestor_name: str
    ingestor_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RunMetrics:
    step: int
    train: dict[str, float]
    validation: dict[str, float]
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BackendInfo:
    name: str
    version: str
    task_types: tuple[str, ...]
    metrics: tuple[str, ...]
    export_formats: tuple[str, ...]
    available: bool = True
    diagnostic: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IngestorInfo:
    name: str
    version: str
    input_formats: tuple[str, ...]
    task_types: tuple[str, ...]
    available: bool = True
    diagnostic: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetInfo:
    id: str
    source: str
    path: str
    records: int
    features: int
    target: str
    classes: list[str]
    format: str = "csv"
    skipped: int = 0
    sha256: str = ""
    size_bytes: int = 0
    task_types: list[str] = field(default_factory=lambda: [TaskType.BINARY_CLASSIFICATION.value])
    schema: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    reused: bool = False
    native_probe: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunConfig:
    dataset: str
    backend: str = "tensorflow"
    objective: str = "binary_classification"
    architecture: str = "auto"
    model_format: str = "auto"
    name: str = "new-classifier"
    mode: str = "Новая модель"
    profile: str = "auto"
    priority: str = "normal"
    target_metric: float = 0.92
    batch_mode: str = "auto"
    batch_size: int | None = None
    max_steps: int = 1400
    cpu: int | None = None
    ram: int | None = None
    gpu: int | None = None
    base_run_id: str | None = None
    model_path: str | None = None
    accept_batch_risk: bool = False
    seed: int = 42
    learning_rate: float | None = None
    weight_decay: float = 0.0
    hidden_layers: tuple[int, ...] = ()
    precision: str = "auto"
    data_workers: int = 0
    prefetch: int = 1
    evaluation_interval: int | None = None
    min_improvement: float = 0.0005
    degradation_margin: float | None = None
    degradation_patience: int = 3
    early_stopping_patience: int = 18
    target_patience: int = 3


@dataclass(slots=True)
class RunInfo:
    id: str
    name: str
    dataset: str
    backend: str
    effective_backend: str | None
    objective: str
    architecture: str
    model_format: str
    mode: str
    profile: str
    priority: str
    target_score: float
    batch_mode: str
    batch_size: int
    batch_min: int
    batch_max: int
    batch_risk: str
    batch_reason: str
    batch_warnings: list[str]
    max_steps: int
    cpu: int
    ram: int
    gpu: int
    base_run_id: str | None = None
    model_path: str | None = None
    seed: int = 42
    learning_rate: float | None = None
    weight_decay: float = 0.0
    hidden_layers: tuple[int, ...] = ()
    precision: str = "auto"
    data_workers: int = 0
    prefetch: int = 1
    evaluation_interval: int | None = None
    min_improvement: float = 0.0005
    degradation_margin: float = 0.03
    degradation_patience: int = 3
    early_stopping_patience: int = 18
    target_patience: int = 3
    status: str = "draft"
    step: int = 0
    best_score: float = 0.0
    best_step: int = 0
    loss: float = 0.0
    samples_seen: int = 0
    eval_count: int = 0
    batch_adjustments: int = 0
    message: str = "Создан; ожидает явного запуска"
    stop_requested: bool = False
    paused: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    termination_reason: str | None = None
    worker_protocol: str | None = None
    worker_pid: int | None = None
    resource_enforcement: dict[str, Any] = field(default_factory=dict)
    environment_manifest: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @classmethod
    def new(cls, **values: Any) -> RunInfo:
        return cls(id=uuid.uuid4().hex[:10], **values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
