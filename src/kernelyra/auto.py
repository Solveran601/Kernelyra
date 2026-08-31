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
from .checkpoints import resolve_checkpoint_policy
from .data_health import analyze_inspection, estimate_records
from .errors import ConfigurationError, DatasetError, RunError
from .hardware import (
    execution_policy,
    resolve_execution_target,
)
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
    "execution": "KERNELYRA_EXECUTION",
    "batch_size": "KERNELYRA_BATCH_SIZE",
    "max_steps": "KERNELYRA_MAX_STEPS",
    "target_metric": "KERNELYRA_TARGET_METRIC",
    "cpu": "KERNELYRA_CPU_PERCENT",
    "ram": "KERNELYRA_RAM_PERCENT",
    "gpu": "KERNELYRA_GPU_PERCENT",
    "threads": "KERNELYRA_THREADS",
    "seed": "KERNELYRA_SEED",
    "learning_rate": "KERNELYRA_LEARNING_RATE",
    "weight_decay": "KERNELYRA_WEIGHT_DECAY",
    "hidden_layers": "KERNELYRA_HIDDEN_LAYERS",
    "precision": "KERNELYRA_PRECISION",
    "data_mode": "KERNELYRA_DATA_MODE",
    "data_workers": "KERNELYRA_DATA_WORKERS",
    "prefetch": "KERNELYRA_PREFETCH",
    "evaluation_interval": "KERNELYRA_EVALUATION_INTERVAL",
    "min_improvement": "KERNELYRA_MIN_IMPROVEMENT",
    "degradation_margin": "KERNELYRA_DEGRADATION_MARGIN",
    "degradation_patience": "KERNELYRA_DEGRADATION_PATIENCE",
    "early_stopping_patience": "KERNELYRA_EARLY_STOPPING_PATIENCE",
    "target_patience": "KERNELYRA_TARGET_PATIENCE",
    "validation_percent": "KERNELYRA_VALIDATION_PERCENT",
    "test_percent": "KERNELYRA_TEST_PERCENT",
    "group_column": "KERNELYRA_GROUP_COLUMN",
    "chunk_target_records": "KERNELYRA_CHUNK_TARGET_RECORDS",
    "chunk_minimum_records": "KERNELYRA_CHUNK_MINIMUM_RECORDS",
    "chunk_maximum_records": "KERNELYRA_CHUNK_MAXIMUM_RECORDS",
    "checkpoint_resume": "KERNELYRA_CHECKPOINT_RESUME",
    "checkpoint_final": "KERNELYRA_CHECKPOINT_FINAL",
    "checkpoint_rollback": "KERNELYRA_CHECKPOINT_ROLLBACK",
}

_DEFAULTS: dict[str, Any] = {
    "target": None,
    "task": "auto",
    "backend": "auto",
    "architecture": "auto",
    "model_format": "auto",
    "execution": "auto",
    "batch_size": None,
    "max_steps": 1400,
    "target_metric": None,
    "cpu": None,
    "ram": None,
    "gpu": None,
    "threads": None,
    "seed": 42,
    "learning_rate": None,
    "weight_decay": 0.0,
    "hidden_layers": None,
    "precision": "auto",
    "data_mode": "auto",
    "data_workers": None,
    "prefetch": None,
    "evaluation_interval": None,
    "min_improvement": 0.0005,
    "degradation_margin": None,
    "degradation_patience": 3,
    "early_stopping_patience": 18,
    "target_patience": 3,
    "validation_percent": 15,
    "test_percent": 15,
    "group_column": None,
    "chunk_target_records": None,
    "chunk_minimum_records": None,
    "chunk_maximum_records": None,
    "checkpoint_resume": "none",
    "checkpoint_final": "none",
    "checkpoint_rollback": "none",
}

_INTEGER_FIELDS = {
    "batch_size", "max_steps", "cpu", "ram", "gpu", "threads", "seed", "data_workers", "prefetch",
    "evaluation_interval", "degradation_patience", "early_stopping_patience", "target_patience",
    "validation_percent", "test_percent", "chunk_target_records", "chunk_minimum_records", "chunk_maximum_records",
}
_FLOAT_FIELDS = {"target_metric", "learning_rate", "weight_decay", "min_improvement", "degradation_margin"}
_REMOVED_OPTION_NAMES = {"algorithm_pack", "profile"}
_REMOVED_ENV_KEYS = {"KERNELYRA_ALGORITHM_PACK", "KERNELYRA_PROFILE"}


def _stream_limit(policy: Mapping[str, Any], maximum: int) -> int:
    return min(maximum, int(policy["stream_limit"]))


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
    if name == "group_column":
        return str(value).strip()
    if name in {"task", "backend", "architecture", "model_format", "execution", "precision", "data_mode", "checkpoint_resume", "checkpoint_final", "checkpoint_rollback"}:
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
    execution: str
    batch_size: int
    max_steps: int
    target_metric: float
    cpu: int
    ram: int
    gpu: int
    threads: int
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
    validation_percent: int
    test_percent: int
    group_column: str | None
    checkpoint_policy: dict[str, str]
    records_estimate: int
    features_estimate: int
    size_bytes: int
    data_mode: str
    config_path: str | None
    sources: dict[str, str]
    data_contract: dict[str, Any]
    split_policy: dict[str, Any]
    chunk_policy: dict[str, Any]
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
        workspace: str | Path | None = None,
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
        removed = sorted(set(configured) & _REMOVED_OPTION_NAMES)
        if removed:
            raise ConfigurationError(
                "Removed training option(s): " + ", ".join(removed) + ". "
                "Use explicit execution and CPU/RAM/GPU/thread limits instead."
            )
        unknown_configured = sorted(set(configured) - set(_DEFAULTS))
        if unknown_configured:
            raise ConfigurationError("Unknown training config option(s): " + ", ".join(unknown_configured))
        active_removed_env = sorted(name for name in _REMOVED_ENV_KEYS if self.environ.get(name, "") != "")
        if active_removed_env:
            raise ConfigurationError(
                "Removed environment option(s): " + ", ".join(active_removed_env) + ". "
                "Use KERNELYRA_EXECUTION and explicit resource environment variables instead."
            )
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

    def _select_backend(self, requested: str, task: str, policy: Mapping[str, Any]) -> str:
        backends = {item["name"]: item for item in self.workspace.capabilities["backends"]}
        if requested != "auto":
            item = backends.get(requested)
            if item is None:
                raise ConfigurationError(f"Backend '{requested}' is not registered")
            if not item.get("available"):
                raise ConfigurationError(str(item.get("diagnostic") or f"Backend '{requested}' is unavailable"))
            if task not in item.get("task_types", []):
                raise ConfigurationError(f"Backend '{requested}' does not support task '{task}'")
            if policy["execution"] not in item.get("execution_targets", ["cpu"]):
                raise ConfigurationError(
                    f"Backend '{requested}' is CPU-only. Hybrid execution requires a compatible PyTorch or TensorFlow GPU backend."
                )
            return requested
        for candidate in tuple(policy["backend_order"]):
            item = backends.get(candidate)
            if item and item.get("available") and task in item.get("task_types", []):
                return candidate
        if policy["execution"] == "hybrid":
            raise ConfigurationError(
                "Hybrid execution requires an installed PyTorch or TensorFlow backend with a usable GPU; "
                "buy or enable a compatible GPU, or use execution='cpu'."
            )
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
        columns = [str(column) for column in (inspection.get("columns") or [])]
        if target is not None and columns and str(target) not in columns:
            preview = ", ".join(columns[:12])
            suffix = ", ..." if len(columns) > 12 else ""
            raise DatasetError(
                f"Target column '{target}' was not found in the dataset. Available columns: {preview}{suffix}"
            )
        task = resolved.values["task"]
        if task == "auto":
            inspected_tasks = inspection.get("task_types") or []
            task = str(inspected_tasks[0]) if len(inspected_tasks) == 1 else self._task_from_inspection(inspection, target)
        if task not in {item.value for item in TaskType}:
            raise ConfigurationError(f"Unknown task '{task}'")
        try:
            execution = resolve_execution_target(resolved.values["execution"], self.workspace.hardware)
        except KeyError as error:
            raise ConfigurationError(str(error)) from None
        policy = execution_policy(
            self.workspace.hardware,
            execution_target=execution,
        )
        backend = self._select_backend(resolved.values["backend"], task, policy)
        architecture, model_format = resolve_training_contract(
            resolved.values["architecture"], resolved.values["model_format"], backend, task
        )
        cpu = int(resolved.values["cpu"] if resolved.values["cpu"] is not None else 70)
        ram = int(resolved.values["ram"] if resolved.values["ram"] is not None else 70)
        gpu = int(resolved.values["gpu"] if resolved.values["gpu"] is not None else (70 if execution == "hybrid" else 0))
        if not 10 <= cpu <= 100:
            raise ConfigurationError("cpu must be between 10 and 100 percent")
        if not 10 <= ram <= 95:
            raise ConfigurationError("ram must be between 10 and 95 percent")
        if not 0 <= gpu <= 100:
            raise ConfigurationError("gpu must be between 0 and 100 percent")
        if execution == "cpu" and gpu:
            raise ConfigurationError(
                "GPU budget was requested for CPU-only execution. Choose execution='hybrid'; "
                "if no accelerator is detected, buy or enable a compatible GPU."
            )
        cpu_threads = max(1, int(self.workspace.hardware.get("cpu_threads") or 1))
        threads = resolved.values["threads"]
        if threads is None:
            threads = max(1, min(cpu_threads, round(cpu_threads * (cpu / 100) * float(policy["native_thread_fraction"]))))
        threads = int(threads)
        if not 1 <= threads <= cpu_threads:
            raise ConfigurationError(f"threads must be between 1 and detected CPU thread count ({cpu_threads})")
        shape = inspection.get("shape") or []
        features = max(1, int(shape[1]) if len(shape) >= 2 else len(columns) - 1)
        size = int(inspection.get("bytes") or source.stat().st_size)
        records = estimate_records(source, inspection)
        requested_batch = resolved.values["batch_size"]
        batch = plan_batch(
            records=records,
            features=features,
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
            workers = min(
                int(policy["data_workers"]),
                max(0, threads - 1),
            )
        prefetch = resolved.values["prefetch"]
        if prefetch is None:
            prefetch = int(policy["prefetch"])
        hidden = resolved.values["hidden_layers"]
        if hidden is None:
            hidden = tuple(policy["hidden_layers"])
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
        validation_percent = int(resolved.values["validation_percent"])
        test_percent = int(resolved.values["test_percent"])
        if not 0 <= validation_percent <= 95 or not 0 <= test_percent <= 95:
            raise ConfigurationError("validation_percent and test_percent must be between 0 and 95")
        if validation_percent + test_percent > 95:
            raise ConfigurationError("validation_percent + test_percent must leave at least 5% for training")
        group_column = resolved.values["group_column"]
        chunk_target_records = resolved.values["chunk_target_records"]
        chunk_minimum_records = resolved.values["chunk_minimum_records"]
        chunk_maximum_records = resolved.values["chunk_maximum_records"]
        try:
            checkpoint_policy = resolve_checkpoint_policy(
                {
                    "resume": resolved.values["checkpoint_resume"],
                    "final": resolved.values["checkpoint_final"],
                    "rollback": resolved.values["checkpoint_rollback"],
                }
            )
        except RunError as error:
            raise ConfigurationError(str(error)) from None
        warnings = list(batch.warnings)
        streaming_formats = {".csv", ".tsv", ".jsonl", ".ndjson", ".parquet", ".pq"}
        # Text tables expand substantially when parsed into Python/NumPy values.
        # A fixed 512 MiB copy limit is therefore unsafe on low-memory machines:
        # the source, decoded rows, encoded arrays and train/validation/test
        # buffers can coexist. Select the streaming path from the resolved
        # resolved RAM limit instead of waiting for the import hard limit.
        stream_limit = _stream_limit(policy, self.workspace.datasets.MAX_IMPORT_BYTES)
        requested_data_mode = resolved.values["data_mode"]
        if requested_data_mode not in {"auto", "memory", "stream"}:
            raise ConfigurationError("data_mode must be auto, memory or stream")
        automatic_data_mode = "stream" if source.is_dir() or size > stream_limit else "memory"
        if requested_data_mode == "memory" and automatic_data_mode == "stream":
            raise ConfigurationError(
                "data_mode='memory' is unsafe for this dataset; use data_mode='stream' or automatic mode"
            )
        data_mode = automatic_data_mode if requested_data_mode == "auto" else requested_data_mode
        if data_mode == "stream" and source.is_file() and source.suffix.lower() not in streaming_formats:
            raise DatasetError(
                f"Dataset exceeds the in-memory limit, but {source.suffix or 'this format'} has no streaming reader"
            )
        if data_mode == "stream":
            origin = "explicit" if requested_data_mode == "stream" else "automatic"
            warnings.append(
                f"Dataset will use the external streaming path ({origin}); the source file must remain available"
            )
        try:
            data_health = analyze_inspection(
                inspection,
                target=str(target) if target is not None else None,
                records_estimate=records,
                feature_count=features,
                seed=int(resolved.values["seed"]),
                chunk_target_records=(
                    int(policy["chunk_target_records"])
                    if chunk_target_records is None
                    else int(chunk_target_records)
                ),
                chunk_minimum_records=(
                    None if chunk_minimum_records is None else int(chunk_minimum_records)
                ),
                chunk_maximum_records=(
                    None if chunk_maximum_records is None else int(chunk_maximum_records)
                ),
                validation_percent=validation_percent,
                test_percent=test_percent,
                group_column=group_column,
                streaming=data_mode == "stream",
            )
        except ValueError as error:
            raise ConfigurationError(str(error)) from None
        data_contract = dict(data_health["contract"])
        split_policy = dict(data_contract["split_policy"])
        chunk_policy = dict(data_contract["chunk_policy"])
        warnings.extend(str(item) for item in data_health["warnings"])
        if split_policy["strategy"] == "context":
            if requested_data_mode == "memory":
                raise ConfigurationError(
                    "data_mode='memory' cannot preserve group/context boundaries; use data_mode='stream' or automatic mode"
                )
            if source.is_dir() or source.suffix.lower() in streaming_formats:
                if data_mode != "stream":
                    data_mode = "stream"
                    warnings.append(
                        "Context-preserving splitting selected the external streaming path so one group cannot cross train, validation and test."
                    )
            else:
                warnings.append(
                    "A context-like column was detected, but this format has no streaming group splitter; do not treat a random split as leakage-safe."
                )
        return TrainingPlan(
            dataset=str(source),
            target=str(target) if target is not None else None,
            task=task,
            backend=backend,
            architecture=architecture,
            model_format=model_format,
            execution=execution,
            batch_size=batch.applied,
            max_steps=max_steps,
            target_metric=target_metric,
            cpu=cpu,
            ram=ram,
            gpu=gpu,
            threads=threads,
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
            validation_percent=validation_percent,
            test_percent=test_percent,
            group_column=group_column,
            checkpoint_policy=checkpoint_policy,
            records_estimate=records,
            features_estimate=features,
            size_bytes=size,
            data_mode=data_mode,
            config_path=str(resolved.config_path) if resolved.config_path else None,
            sources=resolved.sources,
            data_contract=data_contract,
            split_policy=split_policy,
            chunk_policy=chunk_policy,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def train(
        self,
        dataset: str | Path,
        *,
        progress: Callable[[RunInfo], None] | None = None,
        poll_interval: float = .25,
        model: str | Path | None = None,
        base_run_id: str | None = None,
        **overrides: Any,
    ) -> TrainingResult:
        plan = self.plan(dataset, **overrides)
        if plan.data_mode == "stream":
            imported = self.workspace.datasets.attach_path(
                plan.dataset,
                plan.target,
                split_seed=plan.seed,
                validation_percent=plan.validation_percent,
                test_percent=plan.test_percent,
                group_column=plan.group_column,
            )
        else:
            imported = self.workspace.datasets.import_file(plan.dataset, plan.target)
        task = plan.task if plan.task in imported.task_types else imported.task_types[0]
        backend = self._select_backend(
            plan.backend,
            task,
            execution_policy(
                self.workspace.hardware,
                execution_target=plan.execution,
            ),
        )
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
                execution=plan.execution,
                target_metric=plan.target_metric,
                batch_mode="manual" if overrides.get("batch_size") is not None else "auto",
                batch_size=plan.batch_size,
                max_steps=plan.max_steps,
                cpu=plan.cpu,
                ram=plan.ram,
                gpu=plan.gpu,
                threads=plan.threads,
                base_run_id=base_run_id,
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
                checkpoint_policy=plan.checkpoint_policy,
                data_contract=plan.data_contract,
                split_policy=plan.split_policy,
                chunk_policy=plan.chunk_policy,
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
        final_kind = plan.checkpoint_policy["final"]
        checkpoint = (
            self.workspace.runtime.checkpoint_path(current.id, kind=final_kind)
            if final_kind != "none"
            else None
        )
        return TrainingResult(
            plan=plan,
            dataset=imported,
            run=current,
            checkpoint_path=str(checkpoint) if checkpoint is not None and checkpoint.is_file() else None,
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
        base_run_id: str | None = None
        try:
            metadata = self.workspace.runtime.checkpoints.verify(model_path)
            candidate = metadata.get("run_id")
            if isinstance(candidate, str) and self.workspace.storage.get_run(candidate) is not None:
                base_run_id = candidate
        except RunError:
            # External checkpoints remain valid fine-tune inputs.  They have
            # no local run lineage to preserve.
            pass
        return self.train(dataset, model=model_path, base_run_id=base_run_id, **overrides)


def plan(dataset: str | Path, *, workspace: str | Path | None = None, config: str | Path | None = None, **options: Any) -> TrainingPlan:
    with AutoTrainer(workspace, config=config) as trainer:
        return trainer.plan(dataset, **options)


def train(dataset: str | Path, *, workspace: str | Path | None = None, config: str | Path | None = None, **options: Any) -> TrainingResult:
    with AutoTrainer(workspace, config=config) as trainer:
        return trainer.train(dataset, **options)


def finetune(
    model: str | Path,
    dataset: str | Path,
    *,
    workspace: str | Path | None = None,
    config: str | Path | None = None,
    **options: Any,
) -> TrainingResult:
    with AutoTrainer(workspace, config=config) as trainer:
        return trainer.finetune(model, dataset, **options)
