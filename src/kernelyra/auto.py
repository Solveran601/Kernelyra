from __future__ import annotations

import os
import time
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .architectures import resolve_training_contract
from .batch import plan_batch
from .errors import ConfigurationError, DatasetError, RunError
from .hardware import PROFILE_PRESETS, recommend_profile
from .models import DatasetInfo, RunConfig, RunInfo, RunStatus, TaskType
from .workspace import Workspace

TERMINAL_STATES = {
    RunStatus.COMPLETED.value,
    RunStatus.STOPPED.value,
    RunStatus.ERROR.value,
    RunStatus.ERROR_RECOVERABLE.value,
}

_ENV_KEYS = {
    "target": "KERNELYRA_TARGET",
    "task": "KERNELYRA_TASK",
    "backend": "KERNELYRA_BACKEND",
    "architecture": "KERNELYRA_ARCHITECTURE",
    "model_format": "KERNELYRA_MODEL_FORMAT",
    "profile": "KERNELYRA_PROFILE",
    "batch_size": "KERNELYRA_BATCH_SIZE",
    "max_steps": "KERNELYRA_MAX_STEPS",
    "target_metric": "KERNELYRA_TARGET_METRIC",
    "cpu": "KERNELYRA_CPU_PERCENT",
    "ram": "KERNELYRA_RAM_PERCENT",
    "gpu": "KERNELYRA_GPU_PERCENT",
    "seed": "KERNELYRA_SEED",
    "learning_rate": "KERNELYRA_LEARNING_RATE",
    "weight_decay": "KERNELYRA_WEIGHT_DECAY",
    "hidden_layers": "KERNELYRA_HIDDEN_LAYERS",
    "precision": "KERNELYRA_PRECISION",
    "data_workers": "KERNELYRA_DATA_WORKERS",
    "prefetch": "KERNELYRA_PREFETCH",
    "evaluation_interval": "KERNELYRA_EVALUATION_INTERVAL",
    "min_improvement": "KERNELYRA_MIN_IMPROVEMENT",
    "degradation_margin": "KERNELYRA_DEGRADATION_MARGIN",
    "degradation_patience": "KERNELYRA_DEGRADATION_PATIENCE",
    "early_stopping_patience": "KERNELYRA_EARLY_STOPPING_PATIENCE",
    "target_patience": "KERNELYRA_TARGET_PATIENCE",
}

_DEFAULTS: dict[str, Any] = {
    "target": None,
    "task": "auto",
    "backend": "auto",
    "architecture": "auto",
    "model_format": "auto",
    "profile": "auto",
    "batch_size": None,
    "max_steps": 1400,
    "target_metric": None,
    "cpu": None,
    "ram": None,
    "gpu": None,
    "seed": 42,
    "learning_rate": None,
    "weight_decay": 0.0,
    "hidden_layers": None,
    "precision": "auto",
    "data_workers": None,
    "prefetch": None,
    "evaluation_interval": None,
    "min_improvement": 0.0005,
    "degradation_margin": None,
    "degradation_patience": 3,
    "early_stopping_patience": 18,
    "target_patience": 3,
}

_INTEGER_FIELDS = {
    "batch_size", "max_steps", "cpu", "ram", "gpu", "seed", "data_workers", "prefetch",
    "evaluation_interval", "degradation_patience", "early_stopping_patience", "target_patience",
}
_FLOAT_FIELDS = {"target_metric", "learning_rate", "weight_decay", "min_improvement", "degradation_margin"}


def _stream_limit(profile: str, maximum: int) -> int:
    profile_limit = {
        "eco": 96 * 1024 * 1024,
        "low-memory": 128 * 1024 * 1024,
        "balanced": 256 * 1024 * 1024,
        "custom": 256 * 1024 * 1024,
        "performance": 384 * 1024 * 1024,
        "workstation": maximum,
    }[profile]
    return min(maximum, profile_limit)


def _coerce(name: str, value: Any) -> Any:
    if value is None:
        return None
    if name in _INTEGER_FIELDS:
        return int(value)
    if name in _FLOAT_FIELDS:
        return float(value)
    if name == "hidden_layers":
        if isinstance(value, str):
            values = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, list | tuple):
            values = list(value)
        else:
            raise ConfigurationError("hidden_layers must be a comma-separated string or a list")
        result = tuple(int(item) for item in values)
        if any(item < 1 or item > 65_536 for item in result):
            raise ConfigurationError("hidden_layers values must be between 1 and 65536")
        return result
    if name in {"task", "backend", "architecture", "model_format", "profile", "precision"}:
        return str(value).strip().lower()
    return str(value)


def _config_values(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f"Cannot read Kernelyra config: {error}") from None
    section = payload.get("training", payload.get("kernelyra", {}))
    if not isinstance(section, dict):
        raise ConfigurationError("Kernelyra config section [training] must be a table")
    return {str(key): value for key, value in section.items()}


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    dataset: str
    target: str | None
    task: str
    backend: str
    architecture: str
    model_format: str
    profile: str
    batch_size: int
    max_steps: int
    target_metric: float
    cpu: int
    ram: int
    gpu: int
    seed: int
    learning_rate: float | None
    weight_decay: float
    hidden_layers: tuple[int, ...]
    precision: str
    data_workers: int
    prefetch: int
    evaluation_interval: int | None
    min_improvement: float
    degradation_margin: float
    degradation_patience: int
    early_stopping_patience: int
    target_patience: int
    records_estimate: int
    features_estimate: int
    size_bytes: int
    data_mode: str
    config_path: str | None
    sources: dict[str, str]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    plan: TrainingPlan
    dataset: DatasetInfo
    run: RunInfo
    checkpoint_path: str | None = None

    @property
    def checkpoint(self) -> str | None:
        value = self.checkpoint_path or self.run.checkpoint.get("path")
        return str(value) if value else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "dataset": self.dataset.to_dict(),
            "run": self.run.to_dict(),
            "checkpoint": self.checkpoint,
        }


@dataclass(slots=True)
class _Resolved:
    values: dict[str, Any]
    sources: dict[str, str]
    config_path: Path | None


class AutoTrainer:
    """Terminal-first and library-first Kernelyra orchestration surface."""

    def __init__(
        self,
        workspace: str | Path = ".",
        *,
        config: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ):
        self.workspace = Workspace.open(workspace)
        explicit_config = Path(config).expanduser().resolve() if config else None
        default_config = self.workspace.root / "kernelyra.toml"
        self.config_path = explicit_config or (default_config if default_config.exists() else None)
        self.environ = dict(os.environ if environ is None else environ)
        self._closed = False

    def __enter__(self) -> AutoTrainer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> bool:
        if self._closed:
            return True
        self._closed = True
        return self.workspace.close()

    def _resolve(self, explicit: Mapping[str, Any]) -> _Resolved:
        configured = _config_values(self.config_path)
        values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        for name, default in _DEFAULTS.items():
            if explicit.get(name) is not None:
                raw, source = explicit[name], "explicit"
            elif self.environ.get(_ENV_KEYS[name], "") != "":
                raw, source = self.environ[_ENV_KEYS[name]], "environment"
            elif name in configured:
                raw, source = configured[name], "config"
            else:
                raw, source = default, "automatic"
            values[name] = _coerce(name, raw)
            sources[name] = source
        return _Resolved(values, sources, self.config_path)

    @staticmethod
    def _task_from_inspection(inspection: Mapping[str, Any], target: str | None) -> str:
        rows = inspection.get("preview") or []
        values = [str(row.get(target, "")).strip() for row in rows if isinstance(row, dict)] if target else []
        values = [value for value in values if value != ""]
        unique = set(values)
        if len(unique) == 2:
            return TaskType.BINARY_CLASSIFICATION.value
        numeric = True
        for value in values:
            try:
                float(value)
            except ValueError:
                numeric = False
                break
        if unique and (not numeric or len(unique) <= max(20, int(len(values) ** .5) + 1)):
            return TaskType.MULTICLASS_CLASSIFICATION.value
        return TaskType.REGRESSION.value

    def _select_backend(self, requested: str, task: str) -> str:
        backends = {item["name"]: item for item in self.workspace.capabilities["backends"]}
        if requested != "auto":
            item = backends.get(requested)
            if item is None:
                raise ConfigurationError(f"Backend '{requested}' is not registered")
            if not item.get("available"):
                raise ConfigurationError(str(item.get("diagnostic") or f"Backend '{requested}' is unavailable"))
            if task not in item.get("task_types", []):
                raise ConfigurationError(f"Backend '{requested}' does not support task '{task}'")
            return requested
        for candidate in ("native", "torch", "tensorflow", "numpy"):
            item = backends.get(candidate)
            if item and item.get("available") and task in item.get("task_types", []):
                return candidate
        raise ConfigurationError(f"No available backend supports task '{task}'")

    def plan(self, dataset: str | Path, **overrides: Any) -> TrainingPlan:
        unknown = sorted(set(overrides) - set(_DEFAULTS) - {"accept_batch_risk", "name"})
        if unknown:
            raise ConfigurationError(f"Unknown training option(s): {', '.join(unknown)}")
        source = Path(dataset).expanduser().resolve()
        if not source.exists() or not (source.is_file() or source.is_dir()):
            raise DatasetError("Dataset file or folder was not found")
        resolved = self._resolve(overrides)
        inspection = self.workspace.datasets.inspect(source)
        if not inspection.get("trainable"):
            raise DatasetError("Dataset format is recognized but no trainable ingestor is installed")
        target = resolved.values["target"] or inspection.get("suggested_target")
        task = resolved.values["task"]
        if task == "auto":
            inspected_tasks = inspection.get("task_types") or []
            task = str(inspected_tasks[0]) if len(inspected_tasks) == 1 else self._task_from_inspection(inspection, target)
        if task not in {item.value for item in TaskType}:
            raise ConfigurationError(f"Unknown task '{task}'")
        backend = self._select_backend(resolved.values["backend"], task)
        architecture, model_format = resolve_training_contract(
            resolved.values["architecture"], resolved.values["model_format"], backend, task
        )
        profile = resolved.values["profile"]
        if profile == "auto":
            profile = recommend_profile(self.workspace.hardware)
        if profile not in PROFILE_PRESETS:
            raise ConfigurationError(f"Unknown hardware profile '{profile}'")
        preset = PROFILE_PRESETS[profile]
        cpu = int(resolved.values["cpu"] if resolved.values["cpu"] is not None else preset["cpu"])
        ram = int(resolved.values["ram"] if resolved.values["ram"] is not None else preset["ram"])
        gpu = int(
            resolved.values["gpu"]
            if resolved.values["gpu"] is not None
            else preset["gpu"] if self.workspace.hardware["gpu_available"] else 0
        )
        if not 10 <= cpu <= 100:
            raise ConfigurationError("cpu must be between 10 and 100 percent")
        if not 10 <= ram <= 95:
            raise ConfigurationError("ram must be between 10 and 95 percent")
        if not 0 <= gpu <= 100:
            raise ConfigurationError("gpu must be between 0 and 100 percent")
        columns = inspection.get("columns") or []
        shape = inspection.get("shape") or []
        features = max(1, int(shape[1]) if len(shape) >= 2 else len(columns) - 1)
        sampled = max(1, int(inspection.get("sampled_rows") or 1))
        size = int(inspection.get("bytes") or source.stat().st_size)
        preview_bytes = max(1, sum(len(str(row)) for row in inspection.get("preview") or []))
        preview_count = max(1, len(inspection.get("preview") or []))
        known_records = inspection.get("rows") or (shape[0] if len(shape) >= 2 else None)
        records = int(known_records) if known_records else max(sampled, int(size / max(1, preview_bytes / preview_count)))
        requested_batch = resolved.values["batch_size"]
        batch = plan_batch(
            records=records,
            features=features,
            profile=profile,
            ram_percent=ram,
            ram_gb=float(self.workspace.hardware.get("ram_gb") or 8),
            mode="manual" if requested_batch is not None else "auto",
            requested=requested_batch,
        )
        if batch.requires_confirmation and not bool(overrides.get("accept_batch_risk")):
            raise ConfigurationError(
                f"Batch {batch.applied} exceeds safe range {batch.safe_min}-{batch.safe_max}; "
                "use auto batch or accept_batch_risk=True"
            )
        target_metric = resolved.values["target_metric"]
        if target_metric is None:
            target_metric = .80 if task == TaskType.REGRESSION.value else .92
        target_metric = float(target_metric)
        minimum_metric = -10.0 if task == TaskType.REGRESSION.value else 0.0
        if not minimum_metric <= target_metric <= 1.0:
            raise ConfigurationError("target_metric is outside the valid range for this task")
        workers = resolved.values["data_workers"]
        if workers is None:
            workers = max(0, min(8, int(self.workspace.hardware["cpu_threads"]) // 4))
        prefetch = resolved.values["prefetch"]
        if prefetch is None:
            prefetch = 1 if profile in {"eco", "low-memory"} else 2
        hidden = resolved.values["hidden_layers"]
        if hidden is None:
            hidden = {
                "eco": (16, 8),
                "low-memory": (16, 8),
                "balanced": (64, 32),
                "custom": (64, 32),
                "performance": (128, 64, 32),
                "workstation": (256, 128, 64),
            }[profile]
        hidden = tuple(int(width) for width in hidden)
        if not hidden or len(hidden) > 16 or any(not 1 <= width <= 1_000_000 for width in hidden):
            raise ConfigurationError("hidden_layers must contain 1-16 positive widths no larger than 1000000")
        learning_rate = resolved.values["learning_rate"]
        if learning_rate is not None and float(learning_rate) <= 0:
            raise ConfigurationError("learning_rate must be positive")
        if float(resolved.values["weight_decay"]) < 0:
            raise ConfigurationError("weight_decay cannot be negative")
        precision = resolved.values["precision"]
        if precision not in {"auto", "float16", "bfloat16", "float32", "float64"}:
            raise ConfigurationError("precision must be auto, float16, bfloat16, float32 or float64")
        max_steps = int(resolved.values["max_steps"])
        if not 1 <= max_steps <= 10_000_000:
            raise ConfigurationError("max_steps must be between 1 and 10000000")
        workers = int(workers)
        prefetch = int(prefetch)
        if not 0 <= workers <= 64:
            raise ConfigurationError("data_workers must be between 0 and 64")
        if not 0 <= prefetch <= 32:
            raise ConfigurationError("prefetch must be between 0 and 32")
        evaluation_interval = resolved.values["evaluation_interval"]
        if evaluation_interval is not None:
            evaluation_interval = int(evaluation_interval)
            if not 1 <= evaluation_interval <= 1_000_000:
                raise ConfigurationError("evaluation_interval must be between 1 and 1000000")
        min_improvement = float(resolved.values["min_improvement"])
        if not 0.0 <= min_improvement <= 1.0:
            raise ConfigurationError("min_improvement must be between 0 and 1")
        degradation_margin = resolved.values["degradation_margin"]
        if degradation_margin is None:
            degradation_margin = .05 if task == TaskType.REGRESSION.value else .03
        degradation_margin = float(degradation_margin)
        if not 0.0 < degradation_margin <= 10.0:
            raise ConfigurationError("degradation_margin must be greater than 0 and no larger than 10")
        degradation_patience = int(resolved.values["degradation_patience"])
        early_stopping_patience = int(resolved.values["early_stopping_patience"])
        target_patience = int(resolved.values["target_patience"])
        if not 1 <= degradation_patience <= 100:
            raise ConfigurationError("degradation_patience must be between 1 and 100")
        if not 1 <= early_stopping_patience <= 10_000:
            raise ConfigurationError("early_stopping_patience must be between 1 and 10000")
        if not 1 <= target_patience <= 100:
            raise ConfigurationError("target_patience must be between 1 and 100")
        warnings = list(batch.warnings)
        streaming_formats = {".csv", ".tsv", ".jsonl", ".ndjson", ".parquet", ".pq"}
        # Text tables expand substantially when parsed into Python/NumPy values.
        # A fixed 512 MiB copy limit is therefore unsafe on low-memory machines:
        # the source, decoded rows, encoded arrays and train/validation/test
        # buffers can coexist. Select the streaming path from the resolved
        # hardware profile instead of waiting for the import hard limit.
        stream_limit = _stream_limit(profile, self.workspace.datasets.MAX_IMPORT_BYTES)
        data_mode = "stream" if source.is_dir() or size > stream_limit else "memory"
        if data_mode == "stream" and source.is_file() and source.suffix.lower() not in streaming_formats:
            raise DatasetError(
                f"Dataset exceeds the in-memory limit, but {source.suffix or 'this format'} has no streaming reader"
            )
        if data_mode == "stream":
            warnings.append("Dataset will use the external streaming path; the source file must remain available")
        return TrainingPlan(
            dataset=str(source),
            target=str(target) if target is not None else None,
            task=task,
            backend=backend,
            architecture=architecture,
            model_format=model_format,
            profile=profile,
            batch_size=batch.applied,
            max_steps=max_steps,
            target_metric=target_metric,
            cpu=cpu,
            ram=ram,
            gpu=gpu,
            seed=int(resolved.values["seed"]),
            learning_rate=learning_rate,
            weight_decay=float(resolved.values["weight_decay"]),
            hidden_layers=hidden,
            precision=precision,
            data_workers=workers,
            prefetch=prefetch,
            evaluation_interval=evaluation_interval,
            min_improvement=min_improvement,
            degradation_margin=degradation_margin,
            degradation_patience=degradation_patience,
            early_stopping_patience=early_stopping_patience,
            target_patience=target_patience,
            records_estimate=records,
            features_estimate=features,
            size_bytes=size,
            data_mode=data_mode,
            config_path=str(resolved.config_path) if resolved.config_path else None,
            sources=resolved.sources,
            warnings=tuple(warnings),
        )

    def train(
        self,
        dataset: str | Path,
        *,
        progress: Callable[[RunInfo], None] | None = None,
        poll_interval: float = .25,
        model: str | Path | None = None,
        **overrides: Any,
    ) -> TrainingResult:
        plan = self.plan(dataset, **overrides)
        if plan.data_mode == "stream":
            imported = self.workspace.datasets.attach_path(plan.dataset, plan.target)
        else:
            imported = self.workspace.datasets.import_file(plan.dataset, plan.target)
        task = plan.task if plan.task in imported.task_types else imported.task_types[0]
        backend = self._select_backend(plan.backend, task)
        plan = replace(
            plan,
            target=imported.target,
            task=task,
            backend=backend,
            records_estimate=imported.records,
            features_estimate=imported.features,
        )
        run = self.workspace.create_run(
            RunConfig(
                dataset=imported.id,
                backend=backend,
                objective=task,
                architecture=plan.architecture,
                model_format=plan.model_format,
                name=str(overrides.get("name") or Path(plan.dataset).stem)[:80],
                mode="Fine-tune" if model else "Train",
                profile=plan.profile,
                target_metric=plan.target_metric,
                batch_mode="manual" if overrides.get("batch_size") is not None else "auto",
                batch_size=plan.batch_size,
                max_steps=plan.max_steps,
                cpu=plan.cpu,
                ram=plan.ram,
                gpu=plan.gpu,
                model_path=str(Path(model).expanduser().resolve()) if model else None,
                accept_batch_risk=True,
                seed=plan.seed,
                learning_rate=plan.learning_rate,
                weight_decay=plan.weight_decay,
                hidden_layers=plan.hidden_layers,
                precision=plan.precision,
                data_workers=plan.data_workers,
                prefetch=plan.prefetch,
                evaluation_interval=plan.evaluation_interval,
                min_improvement=plan.min_improvement,
                degradation_margin=plan.degradation_margin,
                degradation_patience=plan.degradation_patience,
                early_stopping_patience=plan.early_stopping_patience,
                target_patience=plan.target_patience,
            )
        )
        current = run.start()
        try:
            while current.status not in TERMINAL_STATES:
                if progress:
                    progress(current)
                time.sleep(max(.05, poll_interval))
                current = run.info
        except KeyboardInterrupt:
            run.stop()
            raise
        if progress:
            progress(current)
        if current.status in {RunStatus.ERROR.value, RunStatus.ERROR_RECOVERABLE.value}:
            raise RunError(current.message)
        checkpoint = self.workspace.runtime.checkpoint_path(current.id)
        return TrainingResult(
            plan=plan,
            dataset=imported,
            run=current,
            checkpoint_path=str(checkpoint) if checkpoint.is_file() else None,
        )

    def finetune(
        self,
        model: str | Path,
        dataset: str | Path,
        **overrides: Any,
    ) -> TrainingResult:
        model_path = Path(model).expanduser().resolve()
        if not model_path.is_file():
            raise ConfigurationError("Model file was not found")
        if overrides.get("backend") is None:
            suffix = model_path.suffix.lower()
            overrides["backend"] = (
                "torch" if suffix in {".pt", ".pth"}
                else "tensorflow" if suffix in {".keras", ".h5", ".hdf5"}
                else "numpy" if suffix == ".npz"
                else None
            )
            if overrides["backend"] is None:
                raise ConfigurationError("Cannot infer a backend from the model extension; set backend explicitly")
        return self.train(dataset, model=model_path, **overrides)


def plan(dataset: str | Path, *, workspace: str | Path = ".", config: str | Path | None = None, **options: Any) -> TrainingPlan:
    with AutoTrainer(workspace, config=config) as trainer:
        return trainer.plan(dataset, **options)


def train(dataset: str | Path, *, workspace: str | Path = ".", config: str | Path | None = None, **options: Any) -> TrainingResult:
    with AutoTrainer(workspace, config=config) as trainer:
        return trainer.train(dataset, **options)


def finetune(
    model: str | Path,
    dataset: str | Path,
    *,
    workspace: str | Path = ".",
    config: str | Path | None = None,
    **options: Any,
) -> TrainingResult:
    with AutoTrainer(workspace, config=config) as trainer:
        return trainer.finetune(model, dataset, **options)
