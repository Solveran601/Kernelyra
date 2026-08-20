from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .architectures import resolve_training_contract
from .backends.registry import BackendRegistry
from .batch import BatchPlan, plan_batch
from .capabilities import CapabilityRegistry
from .datasets import DatasetManager
from .errors import ConfigurationError, RunError, RunNotFoundError
from .hardware import PROFILE_PRESETS, detect_hardware, recommend_profile
from .models import RunConfig, RunInfo
from .native_probe import resolve_native_probe
from .storage import SQLiteStorage

if TYPE_CHECKING:
    from .runtime import TrainingRuntime


class Workspace:
    """Explicit Kernelyra workspace. Call ``open`` to create state."""

    def __init__(self, root: Path, state_dir: Path, storage: SQLiteStorage, datasets: DatasetManager):
        self.root = root
        self.state_dir = state_dir
        self.storage = storage
        self.datasets = datasets
        self.hardware = detect_hardware()
        self.backends = BackendRegistry()
        self.capability_registry = CapabilityRegistry(self.backends, self.datasets.ingestors)
        self._runtime: TrainingRuntime | None = None

    @property
    def capabilities(self) -> dict[str, Any]:
        return self.capability_registry.snapshot()

    @classmethod
    def open(cls, path: str | Path) -> Workspace:
        root = Path(path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        state_dir = root / ".kernelyra"
        storage = SQLiteStorage.open(state_dir)
        native_probe = resolve_native_probe(root)
        datasets = DatasetManager(root, state_dir, storage, native_probe)
        return cls(root, state_dir, storage, datasets)

    @property
    def runs(self) -> RunManager:
        return RunManager(self)

    @property
    def runtime(self) -> TrainingRuntime:
        if self._runtime is None:
            from .runtime import TrainingRuntime
            self._runtime = TrainingRuntime(self)
        return self._runtime

    def create_run(self, config: RunConfig) -> RunHandle:
        return self.runs.create(config)

    def inference_check(self, run_id: str, requests: int = 200) -> dict[str, Any]:
        from .inference import run_inference_check

        return run_inference_check(self, run_id, requests)

    def close(self) -> bool:
        if self._runtime:
            return self._runtime.close()
        return True

    def __enter__(self) -> Workspace:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class Kernelyra(Workspace):
    def __init__(self, workspace: str | Path):
        opened = Workspace.open(workspace)
        super().__init__(opened.root, opened.state_dir, opened.storage, opened.datasets)


class RunManager:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def create(self, config: RunConfig) -> RunHandle:
        dataset = self.workspace.datasets.get(config.dataset)
        if config.objective not in dataset.task_types:
            raise ConfigurationError(
                f"Dataset несовместим с task '{config.objective}'. Доступны: {', '.join(dataset.task_types)}"
            )
        backend = next(
            (item for item in self.workspace.backends.describe() if item["name"] == config.backend),
            None,
        )
        if backend is None:
            raise ConfigurationError(f"Backend '{config.backend}' не установлен")
        if config.objective not in backend["task_types"]:
            raise ConfigurationError(f"Backend '{config.backend}' не поддерживает task '{config.objective}'")
        architecture, model_format = resolve_training_contract(
            config.architecture, config.model_format, config.backend, config.objective
        )
        profile = recommend_profile(self.workspace.hardware) if config.profile == "auto" else config.profile
        if profile not in PROFILE_PRESETS:
            raise ConfigurationError("Неизвестный профиль компьютера")
        preset = PROFILE_PRESETS[profile]
        ram = max(10, min(95, int(config.ram if config.ram is not None else preset["ram"])))
        batch = plan_batch(records=dataset.records, features=dataset.features, profile=profile, ram_percent=ram, ram_gb=float(self.workspace.hardware.get("ram_gb") or 8), mode=config.batch_mode, requested=config.batch_size)
        if batch.requires_confirmation and not config.accept_batch_risk:
            raise ConfigurationError(f"Batch {batch.applied} выходит за безопасный диапазон {batch.safe_min}–{batch.safe_max}. Подтвердите риск или включите Auto")
        if config.priority not in {"high", "normal", "low"}:
            raise ConfigurationError("Приоритет должен быть high, normal или low")
        if config.base_run_id and not (self.workspace.state_dir / "checkpoints" / f"{config.base_run_id}.npz").exists():
            raise RunError("Checkpoint исходной модели не найден")
        target_score = (
            max(-10.0, min(.999, config.target_metric))
            if config.objective == "regression"
            else max(.5, min(.999, config.target_metric))
        )
        degradation_margin = config.degradation_margin
        if degradation_margin is None:
            degradation_margin = .05 if config.objective == "regression" else .03
        if config.evaluation_interval is not None and not 1 <= config.evaluation_interval <= 1_000_000:
            raise ConfigurationError("evaluation_interval must be between 1 and 1000000")
        if not 0 <= config.min_improvement <= 1:
            raise ConfigurationError("min_improvement must be between 0 and 1")
        if not 0 < degradation_margin <= 10:
            raise ConfigurationError("degradation_margin must be greater than 0 and no larger than 10")
        if not 1 <= config.degradation_patience <= 100:
            raise ConfigurationError("degradation_patience must be between 1 and 100")
        if not 1 <= config.early_stopping_patience <= 10_000:
            raise ConfigurationError("early_stopping_patience must be between 1 and 10000")
        if not 1 <= config.target_patience <= 100:
            raise ConfigurationError("target_patience must be between 1 and 100")
        run = RunInfo.new(name=config.name[:80], dataset=config.dataset, backend=config.backend, effective_backend=None, objective=config.objective, architecture=architecture, model_format=model_format, mode=config.mode, profile=profile, priority=config.priority, target_score=target_score, batch_mode=config.batch_mode, batch_size=batch.applied, batch_min=batch.safe_min, batch_max=batch.safe_max, batch_risk=batch.risk, batch_reason=batch.reason, batch_warnings=batch.warnings, max_steps=max(1, min(10_000_000, config.max_steps)), cpu=max(10, min(100, int(config.cpu if config.cpu is not None else preset["cpu"]))), ram=ram, gpu=max(0, min(100, int(config.gpu if config.gpu is not None else (preset["gpu"] if self.workspace.hardware["gpu_available"] else 0)))), base_run_id=config.base_run_id, model_path=config.model_path, seed=config.seed, learning_rate=config.learning_rate, weight_decay=max(0.0, float(config.weight_decay)), hidden_layers=tuple(config.hidden_layers), precision=config.precision, data_workers=max(0, min(64, int(config.data_workers))), prefetch=max(0, min(32, int(config.prefetch))), evaluation_interval=config.evaluation_interval, min_improvement=config.min_improvement, degradation_margin=degradation_margin, degradation_patience=config.degradation_patience, early_stopping_patience=config.early_stopping_patience, target_patience=config.target_patience)
        self.workspace.storage.save_run(run)
        self.workspace.storage.log_action("sdk", "run.create", {"run_id": run.id, "dataset": run.dataset, "backend": run.backend})
        return RunHandle(self.workspace, run.id)

    def get(self, run_id: str) -> RunHandle:
        if not self.workspace.storage.get_run(run_id):
            raise RunNotFoundError("Run не найден")
        return RunHandle(self.workspace, run_id)

    def list(self) -> list[RunInfo]:
        return self.workspace.storage.list_runs()

    def remove(self, run_id: str) -> None:
        run = self.get(run_id).info
        if run.status not in {"completed", "stopped", "error", "error_recoverable"}:
            raise RunError("Only terminal runs can be removed")
        self.workspace.runtime.checkpoints.remove_run(run_id)
        self.workspace.storage.delete_run(run_id)
        self.workspace.storage.log_action("sdk", "run.remove", {"run_id": run_id})


class RunHandle:
    def __init__(self, workspace: Workspace, run_id: str):
        self.workspace, self.id = workspace, run_id

    @property
    def info(self) -> RunInfo:
        run = self.workspace.storage.get_run(self.id)
        if not run:
            raise RunNotFoundError("Run не найден")
        return run

    def start(self) -> RunInfo:
        return self.workspace.runtime.command(self.id, "start")

    def pause(self) -> RunInfo:
        return self.workspace.runtime.command(self.id, "pause")

    def resume(self) -> RunInfo:
        return self.workspace.runtime.command(self.id, "resume")

    def stop(self) -> RunInfo:
        return self.workspace.runtime.command(self.id, "stop")


def batch_plan_for(workspace: Workspace, dataset_id: str, profile: str, mode: str, requested: int | None, ram: int) -> BatchPlan:
    dataset = workspace.datasets.get(dataset_id)
    resolved = recommend_profile(workspace.hardware) if profile == "auto" else profile
    return plan_batch(records=dataset.records, features=dataset.features, profile=resolved, ram_percent=ram, ram_gb=float(workspace.hardware.get("ram_gb") or 8), mode=mode, requested=requested)
