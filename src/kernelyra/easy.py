"""Small, stable convenience API shared conceptually by every Kernelyra SDK."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Self

from .auto import AutoTrainer, TrainingPlan, TrainingResult


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

    def profile(self, name: str) -> Self:
        return self.set(profile=name)

    def hardware(
        self,
        profile: str = "auto",
        *,
        cpu: int | None = None,
        ram: int | None = None,
        gpu: int | None = None,
    ) -> Self:
        """Select an automatic/preset profile and optionally override its limits."""
        return self.set(profile=profile, cpu=cpu, ram=ram, gpu=gpu)

    def low_memory(self) -> Self:
        return self.profile("low-memory")

    def weak(self) -> Self:
        """Use the conservative program for a weak PC."""
        return self.low_memory()

    def balanced(self) -> Self:
        return self.profile("balanced")

    def medium(self) -> Self:
        """Alias for the balanced program."""
        return self.balanced()

    def performance(self) -> Self:
        return self.profile("performance")

    def powerful(self) -> Self:
        """Use the throughput-oriented program for a powerful PC."""
        return self.performance()

    def workstation(self) -> Self:
        return self.profile("workstation")

    def custom(self, *, cpu: int, ram: int, gpu: int = 0) -> Self:
        return self.hardware("custom", cpu=cpu, ram=ram, gpu=gpu)

    def goal(self, metric: float) -> Self:
        return self.set(target_metric=metric)

    def steps(self, maximum: int) -> Self:
        return self.set(max_steps=maximum)

    def batch(self, size: int | None = None, *, accept_risk: bool = False) -> Self:
        if size is None:
            return self.unset("batch_size", "accept_batch_risk")
        return self.set(batch_size=size, accept_batch_risk=accept_risk)

    def resources(self, *, cpu: int | None = None, ram: int | None = None, gpu: int | None = None) -> Self:
        return self.set(cpu=cpu, ram=ram, gpu=gpu)

    def optimizer(self, *, learning_rate: float | None = None, weight_decay: float | None = None) -> Self:
        return self.set(learning_rate=learning_rate, weight_decay=weight_decay)

    def model(self, *hidden_layers: int, precision: str | None = None) -> Self:
        values: dict[str, Any] = {}
        if hidden_layers:
            values["hidden_layers"] = tuple(hidden_layers)
        if precision is not None:
            values["precision"] = precision
        return self.set(**values)

    def data(self, *, workers: int | None = None, prefetch: int | None = None) -> Self:
        return self.set(data_workers=workers, prefetch=prefetch)

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
        """Tune Model Guard sensitivity without disabling best-checkpoint protection."""
        return self.set(degradation_margin=margin, degradation_patience=patience)

    def seed(self, value: int) -> Self:
        return self.set(seed=value)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._values)


Config = TrainingConfig
Settings = TrainingConfig


class Engine:
    """Easy library facade: create once, then plan, train or fine-tune."""

    def __init__(
        self,
        workspace: str | Path = ".",
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
    workspace: str | Path = ".",
    config: str | Path | None = None,
    settings: TrainingConfig | Mapping[str, Any] | None = None,
    **options: Any,
) -> TrainingResult:
    """One-call training. Dataset, target and workspace are the only common inputs."""
    with Engine(workspace, config=config) as engine:
        return engine.fit(dataset, target, settings=settings, **options)
