"""Small, stable convenience API shared conceptually by every Kernelyra SDK."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from .auto import _DEFAULTS, AutoTrainer, TrainingPlan, TrainingResult, _coerce
from .checkpoints import resolve_checkpoint_policy
from .errors import ConfigurationError, RunError
from .reports import build_experiment_report, write_experiment_report

if TYPE_CHECKING:
    from .workspace import RunHandle


class TrainingConfig:
    """Fluent training options; every unspecified value remains automatic."""

    __slots__ = ("_values",)

    def __init__(self, **options: Any):
        self._values: dict[str, Any] = {}
        self.set(**options)

    def set(self, **options: Any) -> Self:
        """Set advanced engine options by their stable protocol names."""
        self._values.update({key: value for key, value in options.items() if value is not None})
        return self

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> Self:
        """Create settings from a regular mapping without retaining a reference to it."""
        return cls(**dict(options))

    def copy(self) -> Self:
        """Return an independent configuration that can be changed safely."""
        return type(self)(**deepcopy(self._values))

    def merge(self, *settings: TrainingConfig | Mapping[str, Any], **options: Any) -> Self:
        """Merge settings left-to-right; later explicit values have priority."""
        for item in settings:
            values = item.to_dict() if isinstance(item, TrainingConfig) else dict(item)
            self.set(**values)
        return self.set(**options)

    def unset(self, *names: str) -> Self:
        """Return named options to automatic resolution."""
        for name in names:
            self._values.pop(name, None)
        return self

    def automatic(self, *names: str) -> Self:
        """Reset selected options, or every option when no names are supplied."""
        if names:
            return self.unset(*names)
        self._values.clear()
        return self

    def target(self, column: str) -> Self:
        return self.set(target=column)

    def task(self, name: str) -> Self:
        return self.set(task=name)

    def backend(self, name: str) -> Self:
        return self.set(backend=name)

    def architecture(self, name: str) -> Self:
        return self.set(architecture=name)

    def model_format(self, name: str) -> Self:
        return self.set(model_format=name)

    def execution(self, target: str) -> Self:
        """Choose ``cpu`` or ``hybrid`` execution explicitly."""
        return self.set(execution=target)

    def cpu_only(self) -> Self:
        return self.execution("cpu")

    def hybrid(self) -> Self:
        return self.execution("hybrid")

    def hardware(
        self,
        *,
        execution: str = "auto",
        cpu: int | None = None,
        ram: int | None = None,
        gpu: int | None = None,
        threads: int | None = None,
    ) -> Self:
        """Set execution and explicit CPU, RAM, GPU and thread ceilings."""
        return self.set(execution=execution, cpu=cpu, ram=ram, gpu=gpu, threads=threads)

    def goal(self, metric: float) -> Self:
        return self.set(target_metric=metric)

    def steps(self, maximum: int) -> Self:
        return self.set(max_steps=maximum)

    def batch(self, size: int | None = None, *, accept_risk: bool = False) -> Self:
        if size is None:
            return self.unset("batch_size", "accept_batch_risk")
        return self.set(batch_size=size, accept_batch_risk=accept_risk)

    def resources(
        self, *, cpu: int | None = None, ram: int | None = None, gpu: int | None = None, threads: int | None = None
    ) -> Self:
        """Set resource ceilings as percentages of detected hardware and CPU threads."""
        return self.set(cpu=cpu, ram=ram, gpu=gpu, threads=threads)

    def budget(
        self,
        *,
        cpu_percent: int | None = None,
        memory_percent: int | None = None,
        gpu_percent: int | None = None,
        threads: int | None = None,
    ) -> Self:
        """Set the same explicit resource ceilings with self-documenting names.

        Every omitted value remains automatic, so a caller can cap only RAM or
        CPU without accidentally fixing the rest of the execution plan.
        """
        return self.resources(cpu=cpu_percent, ram=memory_percent, gpu=gpu_percent, threads=threads)

    def optimizer(self, *, learning_rate: float | None = None, weight_decay: float | None = None) -> Self:
        return self.set(learning_rate=learning_rate, weight_decay=weight_decay)

    def model(self, *hidden_layers: int, precision: str | None = None) -> Self:
        values: dict[str, Any] = {}
        if hidden_layers:
            values["hidden_layers"] = tuple(hidden_layers)
        if precision is not None:
            values["precision"] = precision
        return self.set(**values)

    def data(
        self,
        *,
        mode: str | None = None,
        workers: int | None = None,
        prefetch: int | None = None,
    ) -> Self:
        """Choose automatic, in-memory or external-streaming input handling.

        ``memory`` is the fastest path when the complete dataset fits inside
        Kernelyra's safety limit.  ``stream`` keeps RAM bounded and preserves
        context-group splits for formats with a streaming reader.  ``auto``
        retains the planner's safe recommendation.
        """
        return self.set(data_mode=mode, data_workers=workers, prefetch=prefetch)

    def split(
        self,
        *,
        validation_percent: int | None = None,
        test_percent: int | None = None,
        group_column: str | None = None,
    ) -> Self:
        """Control held-out proportions and optionally declare a context key.

        A declared ``group_column`` is excluded from learned features and is
        kept entirely in one split on the external streaming path.  Leaving a
        value omitted keeps the corresponding automatic setting.
        """
        return self.set(
            validation_percent=validation_percent,
            test_percent=test_percent,
            group_column=group_column,
        )

    def chunks(
        self,
        *,
        target_records: int | None = None,
        minimum_records: int | None = None,
        maximum_records: int | None = None,
    ) -> Self:
        """Tune non-uniform contiguous planning ranges without breaking order."""
        return self.set(
            chunk_target_records=target_records,
            chunk_minimum_records=minimum_records,
            chunk_maximum_records=maximum_records,
        )

    def stopping(
        self,
        *,
        maximum_steps: int | None = None,
        target_metric: float | None = None,
        early_stopping_patience: int | None = None,
        target_patience: int | None = None,
    ) -> Self:
        """Configure result-driven stopping and its emergency step ceiling."""
        return self.set(
            max_steps=maximum_steps,
            target_metric=target_metric,
            early_stopping_patience=early_stopping_patience,
            target_patience=target_patience,
        )

    def quality(
        self,
        *,
        evaluation_interval: int | None = None,
        min_improvement: float | None = None,
        early_stopping_patience: int | None = None,
        target_patience: int | None = None,
    ) -> Self:
        """Configure validation cadence and result-driven stopping."""
        return self.set(
            evaluation_interval=evaluation_interval,
            min_improvement=min_improvement,
            early_stopping_patience=early_stopping_patience,
            target_patience=target_patience,
        )

    def guard(self, *, margin: float | None = None, patience: int | None = None) -> Self:
        """Tune Model Guard sensitivity; durable rollback is an explicit checkpoint choice."""
        return self.set(degradation_margin=margin, degradation_patience=patience)

    def checkpoints(
        self,
        *,
        resume: str | None = None,
        final: str | None = None,
        rollback: str | None = None,
    ) -> Self:
        """Choose durable checkpoint roles for a run.

        ``resume`` and ``final`` accept ``"none"``, ``"last"`` or ``"best"``.
        ``rollback`` accepts ``"none"`` or ``"best"``.  The default is no
        persistent checkpoint at all; enabling rollback deliberately stores a
        best checkpoint so Model Guard can restore it.
        """
        values = {
            "checkpoint_resume": resume,
            "checkpoint_final": final,
            "checkpoint_rollback": rollback,
        }
        return self.set(**{key: value for key, value in values.items() if value is not None})

    def seed(self, value: int) -> Self:
        return self.set(seed=value)

    def validate(self) -> dict[str, Any]:
        """Validate explicit settings before a dataset or workspace is opened.

        This checks option names and values that do not depend on the dataset
        or detected hardware.  ``Engine.plan`` remains the final validation
        step because it alone can validate a target column, a backend and a
        resource budget against real inputs.
        """
        unknown = sorted(set(self._values) - set(_DEFAULTS) - {"accept_batch_risk", "name"})
        if unknown:
            raise ConfigurationError("Unknown training option(s): " + ", ".join(unknown))
        values = {
            name: _coerce(name, value)
            for name, value in self._values.items()
            if value is not None
        }
        if values.get("task", "auto") not in {
            "auto", "binary_classification", "multiclass_classification", "regression",
        }:
            raise ConfigurationError("task must be auto, binary_classification, multiclass_classification or regression")
        if values.get("backend", "auto") not in {"auto", "native", "numpy", "torch", "tensorflow"}:
            raise ConfigurationError("backend must be auto, native, numpy, torch or tensorflow")
        if values.get("execution", "auto") not in {"auto", "cpu", "hybrid"}:
            raise ConfigurationError("execution must be auto, cpu or hybrid")
        if values.get("data_mode", "auto") not in {"auto", "memory", "stream"}:
            raise ConfigurationError("data_mode must be auto, memory or stream")
        if values.get("precision", "auto") not in {"auto", "float16", "bfloat16", "float32", "float64"}:
            raise ConfigurationError("precision must be auto, float16, bfloat16, float32 or float64")
        ranges = {
            "cpu": (10, 100),
            "ram": (10, 95),
            "gpu": (0, 100),
            "threads": (1, None),
            "batch_size": (1, None),
            "max_steps": (1, 10_000_000),
            "data_workers": (0, 64),
            "prefetch": (0, 32),
            "evaluation_interval": (1, 1_000_000),
            "validation_percent": (0, 95),
            "test_percent": (0, 95),
            "degradation_patience": (1, 100),
            "early_stopping_patience": (1, 10_000),
            "target_patience": (1, 100),
        }
        for name, (minimum, maximum) in ranges.items():
            if name not in values:
                continue
            value = int(values[name])
            if value < minimum or (maximum is not None and value > maximum):
                ceiling = "unbounded" if maximum is None else str(maximum)
                raise ConfigurationError(f"{name} must be between {minimum} and {ceiling}")
        if int(values.get("validation_percent", 0)) + int(values.get("test_percent", 0)) > 95:
            raise ConfigurationError("validation_percent + test_percent must leave at least 5% for training")
        if "learning_rate" in values and float(values["learning_rate"]) <= 0:
            raise ConfigurationError("learning_rate must be positive")
        if "weight_decay" in values and float(values["weight_decay"]) < 0:
            raise ConfigurationError("weight_decay cannot be negative")
        if "min_improvement" in values and not 0.0 <= float(values["min_improvement"]) <= 1.0:
            raise ConfigurationError("min_improvement must be between 0 and 1")
        if "degradation_margin" in values and not 0.0 < float(values["degradation_margin"]) <= 10.0:
            raise ConfigurationError("degradation_margin must be greater than 0 and no larger than 10")
        chunk_names = ("chunk_minimum_records", "chunk_target_records", "chunk_maximum_records")
        supplied_chunks = {name: int(values[name]) for name in chunk_names if name in values}
        if any(value < 1 for value in supplied_chunks.values()):
            raise ConfigurationError("chunk record counts must be positive")
        lower = supplied_chunks.get("chunk_minimum_records")
        target = supplied_chunks.get("chunk_target_records")
        upper = supplied_chunks.get("chunk_maximum_records")
        if lower is not None and target is not None and lower > target:
            raise ConfigurationError("chunk_minimum_records cannot exceed chunk_target_records")
        if target is not None and upper is not None and target > upper:
            raise ConfigurationError("chunk_target_records cannot exceed chunk_maximum_records")
        try:
            resolve_checkpoint_policy(
                {
                    "resume": values.get("checkpoint_resume", "none"),
                    "final": values.get("checkpoint_final", "none"),
                    "rollback": values.get("checkpoint_rollback", "none"),
                }
            )
        except RunError as error:
            raise ConfigurationError(str(error)) from None
        return dict(values)

    def explain(self) -> dict[str, Any]:
        """Show the explicit/automatic boundary before planning or training.

        No workspace, dataset or file is created by this method.
        """
        explicit = self.validate()
        policy = resolve_checkpoint_policy(
            {
                "resume": explicit.get("checkpoint_resume", "none"),
                "final": explicit.get("checkpoint_final", "none"),
                "rollback": explicit.get("checkpoint_rollback", "none"),
            }
        )
        automatic = sorted(set(_DEFAULTS) - set(explicit))
        return {
            "explicit": explicit,
            "automatic": automatic,
            "checkpoint_policy": policy,
            "persistent_checkpoint_writes": any(value != "none" for value in policy.values()),
            "contract": "Settings.explain performs no filesystem writes; Engine.plan validates dataset and hardware.",
        }

    def to_dict(self) -> dict[str, Any]:
        return dict(self._values)


Config = TrainingConfig
Settings = TrainingConfig


class Engine:
    """Easy library facade with a caller-selected persistent workspace."""

    def __init__(
        self,
        workspace: str | Path | None = None,
        *,
        config: str | Path | None = None,
        settings: TrainingConfig | Mapping[str, Any] | None = None,
    ):
        self._trainer = AutoTrainer(workspace, config=config)
        self._defaults = self._config_values(settings)

    @staticmethod
    def _config_values(settings: TrainingConfig | Mapping[str, Any] | None) -> dict[str, Any]:
        if settings is None:
            return {}
        return settings.to_dict() if isinstance(settings, TrainingConfig) else dict(settings)

    def configure(self, settings: TrainingConfig | Mapping[str, Any] | None = None, **options: Any) -> Self:
        """Update defaults shared by subsequent plan/fit/fine-tune calls."""
        self._defaults.update(self._config_values(settings))
        self._defaults.update({key: value for key, value in options.items() if value is not None})
        return self

    @property
    def hardware(self) -> dict[str, Any]:
        return deepcopy(self._trainer.workspace.hardware)

    @property
    def capabilities(self) -> dict[str, Any]:
        return deepcopy(self._trainer.workspace.capabilities)

    def inspect(self, dataset: str | Path) -> dict[str, Any]:
        """Inspect a file or folder without importing or training it."""
        return self._trainer.workspace.datasets.inspect(Path(dataset).expanduser().resolve())

    def doctor(self, dataset: str | Path, target: str | None = None) -> dict[str, Any]:
        """Run bounded data-health checks and create a dataset contract."""
        return self._trainer.workspace.datasets.doctor(Path(dataset).expanduser().resolve(), target)

    def report(self, run_id: str, output: str | Path | None = None) -> dict[str, Any]:
        """Build a portable experiment report and optionally write it to disk."""
        report = build_experiment_report(self._trainer.workspace, run_id)
        if output is not None:
            report["output"] = str(write_experiment_report(report, output))
        return report

    def run(self, run_id: str) -> RunHandle:
        """Return a safe handle for one run's state, metrics, logs, and report."""
        return self._trainer.workspace.runs.get(run_id)

    def _options(
        self,
        target: str | None,
        settings: TrainingConfig | Mapping[str, Any] | None,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(self._defaults)
        merged.update(self._config_values(settings))
        merged.update({key: value for key, value in options.items() if value is not None})
        if target is not None:
            merged["target"] = target
        return merged

    def plan(
        self,
        dataset: str | Path,
        target: str | None = None,
        *,
        settings: TrainingConfig | Mapping[str, Any] | None = None,
        **options: Any,
    ) -> TrainingPlan:
        return self._trainer.plan(dataset, **self._options(target, settings, options))

    def fit(
        self,
        dataset: str | Path,
        target: str | None = None,
        *,
        settings: TrainingConfig | Mapping[str, Any] | None = None,
        **options: Any,
    ) -> TrainingResult:
        return self._trainer.train(dataset, **self._options(target, settings, options))

    train = fit

    def finetune(
        self,
        model: str | Path,
        dataset: str | Path,
        target: str | None = None,
        *,
        settings: TrainingConfig | Mapping[str, Any] | None = None,
        **options: Any,
    ) -> TrainingResult:
        return self._trainer.finetune(model, dataset, **self._options(target, settings, options))

    def plan_many(
        self,
        datasets: Iterable[str | Path],
        target: str | None = None,
        *,
        settings: TrainingConfig | Mapping[str, Any] | None = None,
        **options: Any,
    ) -> list[TrainingPlan]:
        """Plan several independent files or folders with one shared policy."""
        return [self.plan(dataset, target, settings=settings, **options) for dataset in datasets]

    def fit_many(
        self,
        datasets: Iterable[str | Path],
        target: str | None = None,
        *,
        settings: TrainingConfig | Mapping[str, Any] | None = None,
        **options: Any,
    ) -> list[TrainingResult]:
        """Train several datasets sequentially so resource limits stay enforceable."""
        return [self.fit(dataset, target, settings=settings, **options) for dataset in datasets]

    def close(self) -> bool:
        return self._trainer.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def fit(
    dataset: str | Path,
    target: str | None = None,
    *,
    workspace: str | Path | None = None,
    config: str | Path | None = None,
    settings: TrainingConfig | Mapping[str, Any] | None = None,
    **options: Any,
) -> TrainingResult:
    """One-call training. Dataset, target and workspace are the only common inputs."""
    with Engine(workspace, config=config) as engine:
        return engine.fit(dataset, target, settings=settings, **options)
