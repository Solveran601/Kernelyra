from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any

from .backends import WORKER_PROTOCOL_VERSION, BackendConfig, BackendWorker, ProcessBackendWorker
from .checkpoints import CheckpointManager
from .errors import ConfigurationError, RunError, RunStateError
from .hardware import PROFILE_PRESETS, recommend_profile
from .models import RunInfo
from .storage import SQLiteStorage


class LifecycleOutcome(str, Enum):
    PAUSE = "pause"
    STOP = "stop"
    COMPLETE = "complete"
    ERROR = "error"


class TrainingRuntime:
    """Explicit local scheduler shared by SDK, CLI and HTTP server."""

    GPU_CAP, RAM_CAP, CPU_CAP = 100, 100, 100

    def __init__(self, workspace: Any):
        self.workspace = workspace
        self.storage: SQLiteStorage = workspace.storage
        self.backends = workspace.backends
        self.lock = threading.RLock()
        self.threads: dict[str, threading.Thread] = {}
        self.controls: dict[str, dict[str, bool]] = {}
        self.checkpoints = CheckpointManager(self.workspace.state_dir / "checkpoints")
        self._scheduler: threading.Thread | None = None
        self._closing = threading.Event()
        self._scheduler_lock_handle: Any | None = None

    def _acquire_scheduler_lock(self) -> None:
        if self._scheduler_lock_handle is not None:
            return
        path = self.workspace.state_dir / "scheduler.lock"
        handle = path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                getattr(fcntl, "flock")(
                    handle.fileno(), getattr(fcntl, "LOCK_EX") | getattr(fcntl, "LOCK_NB")
                )
        except OSError as exc:
            handle.close()
            raise ConfigurationError(
                "Для этого workspace уже работает другой TrainingRuntime. Используйте DaemonClient."
            ) from exc
        self._scheduler_lock_handle = handle

    def _release_scheduler_lock(self) -> None:
        handle = self._scheduler_lock_handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                getattr(fcntl, "flock")(handle.fileno(), getattr(fcntl, "LOCK_UN"))
        finally:
            handle.close()
            self._scheduler_lock_handle = None

    def start(self) -> None:
        with self.lock:
            if self._scheduler and self._scheduler.is_alive():
                return
            self._acquire_scheduler_lock()
            for run in self.storage.list_runs():
                if run.status in {"training", "queued", "pausing", "stopping"}:
                    run.status = "error_recoverable"
                    run.paused = True
                    run.termination_reason = "daemon_restart"
                    run.worker_pid = None
                    run.message = "Runtime перезапущен во время работы; checkpoint можно продолжить"
                    self.storage.save_run(run)
            self._closing.clear()
            self._scheduler = threading.Thread(target=self._schedule_forever, name="kernelyra-scheduler", daemon=True)
            self._scheduler.start()

    def close(self, timeout: float = 5.0) -> bool:
        self._closing.set()
        with self.lock:
            for run_id in list(self.threads):
                self.controls.setdefault(run_id, {"pause": False, "stop": False})["pause"] = True
                run = self.storage.get_run(run_id)
                if run and run.status == "training":
                    run.status = "pausing"
                    run.paused = True
                    run.message = "Runtime закрывается; безопасная пауза после batch"
                    self.storage.save_run(run)
            threads = list(self.threads.values())
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        scheduler = self._scheduler
        if scheduler:
            scheduler.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            self.storage.log_action(
                "runtime",
                "shutdown.blocked",
                {"alive_workers": alive, "lock_released": False},
            )
            return False
        self._release_scheduler_lock()
        return True

    def checkpoint_path(self, run_id: str) -> Path:
        return self.checkpoints.best_path(run_id)

    def command(self, run_id: str, command: str, actor: str = "runtime") -> RunInfo:
        with self.lock:
            run = self.storage.get_run(run_id)
            if not run:
                raise RunError("Запуск не найден")
            if command == "start" and run.status in {"queued", "training"}:
                self.storage.log_action(actor, "run.start.idempotent", {"run_id": run.id, "status": run.status})
                return run
            if command == "start":
                if run.status != "draft":
                    raise RunStateError(run.id, command, run.status, ("draft",))
                run.stop_requested = False
                run.paused = False
                run.status = "queued"
                run.message = "Ожидает безопасный ресурсный слот"
                self.controls[run.id] = {"pause": False, "stop": False}
                self.start()
            elif command == "resume":
                if run.status not in {"paused", "stopped", "error_recoverable"}:
                    raise RunStateError(run.id, command, run.status, ("paused", "stopped", "error_recoverable"))
                run.stop_requested = False
                run.paused = False
                run.status = "queued"
                run.message = "Продолжение поставлено в очередь"
                self.controls[run.id] = {"pause": False, "stop": False}
                self.start()
            elif command == "pause":
                if run.status in {"pausing", "paused"}:
                    return run
                if run.status not in {"queued", "training"}:
                    raise RunStateError(run.id, command, run.status, ("queued", "training"))
                worker_alive = bool(self.threads.get(run.id) and self.threads[run.id].is_alive())
                run.paused = True
                run.status = "pausing" if worker_alive else "paused"
                run.message = "Пауза запрошена; завершается текущий batch" if worker_alive else "Поставлено на паузу"
                self.controls.setdefault(run.id, {"pause": False, "stop": False})["pause"] = True
            elif command == "stop":
                if run.status in {"stopping", "stopped"}:
                    return run
                allowed = ("draft", "queued", "training", "pausing", "paused", "error_recoverable")
                if run.status not in allowed:
                    raise RunStateError(run.id, command, run.status, allowed)
                worker_alive = bool(self.threads.get(run.id) and self.threads[run.id].is_alive())
                run.stop_requested = True
                run.paused = True
                run.status = "stopping" if worker_alive else "stopped"
                run.message = "Остановка запрошена; завершается текущий batch" if worker_alive else "Остановлено"
                self.controls.setdefault(run.id, {"pause": False, "stop": False})["stop"] = True
            else:
                raise RunError("Неизвестная команда")
            self.storage.save_run(run)
            self.storage.log_action(actor, f"run.{command}", {"run_id": run.id, "status": run.status})
            return run

    def usage(self) -> dict[str, int]:
        with self.lock:
            live_run_ids = {run_id for run_id, thread in self.threads.items() if thread.is_alive()}
        active = [run for run in self.storage.list_runs() if run.id in live_run_ids]
        return {"gpu": sum(run.gpu for run in active), "ram": sum(run.ram for run in active), "cpu": sum(run.cpu for run in active)}

    def can_start(self, run: RunInfo) -> bool:
        with self.lock:
            live_ids = {run_id for run_id, thread in self.threads.items() if thread.is_alive()}
        for live_id in live_ids:
            active = self.storage.get_run(live_id)
            if active and active.status in {"pausing", "stopping"}:
                return False
        usage = self.usage()
        return usage["gpu"] + run.gpu <= self.GPU_CAP and usage["ram"] + run.ram <= self.RAM_CAP and usage["cpu"] + run.cpu <= self.CPU_CAP

    def _schedule_forever(self) -> None:
        priority = {"high": 0, "normal": 1, "low": 2}
        while not self._closing.wait(.25):
            with self.lock:
                for run in sorted(self.storage.list_runs(), key=lambda item: (priority.get(item.priority, 1), item.created_at)):
                    if run.status != "queued" or run.paused or run.stop_requested or run.id in self.threads:
                        continue
                    if not self.can_start(run):
                        continue
                    run.status = "training"
                    run.message = f"Подготовка backend {run.backend} и адаптивных batch"
                    self.controls.setdefault(run.id, {"pause": False, "stop": False})
                    self.storage.save_run(run)
                    thread = threading.Thread(target=self._train_guarded, args=(run.id,), name=f"kernelyra-run-{run.id}", daemon=True)
                    self.threads[run.id] = thread
                    thread.start()

    def _train_guarded(self, run_id: str) -> None:
        outcome = LifecycleOutcome.ERROR
        failure: Exception | None = None
        try:
            outcome = self._train(run_id)
        except Exception as error:
            failure = error
        finally:
            with self.lock:
                run = self.storage.get_run(run_id)
                control = dict(self.controls.get(run_id, {"pause": False, "stop": False}))
                if failure is None and run:
                    if run.status == "stopping" or control["stop"]:
                        outcome = LifecycleOutcome.STOP
                    elif run.status == "pausing" or control["pause"]:
                        outcome = LifecycleOutcome.PAUSE
                if run:
                    self._finalize_lifecycle_locked(run, outcome, failure)
                self.threads.pop(run_id, None)
                self.controls.pop(run_id, None)

    def _finalize_lifecycle_locked(
        self,
        run: RunInfo,
        outcome: LifecycleOutcome,
        failure: Exception | None,
    ) -> None:
        """Persist the terminal state while the worker thread is still locked/registered."""
        if outcome is LifecycleOutcome.ERROR:
            run.status = "error_recoverable"
            run.paused = True
            run.termination_reason = "worker_crash" if failure else "recoverable_backend_error"
            detail = failure or RuntimeError("unknown worker error")
            run.message = f"Ошибка {type(detail).__name__}: {str(detail)[:150]}. Лучший checkpoint сохранён; исправьте причину и продолжите."
            self.storage.log_action("runtime", "run.error", {"run_id": run.id, "error": str(detail)[:300]})
        elif outcome is LifecycleOutcome.STOP:
            run.status = "stopped"
            run.stop_requested = True
            run.paused = True
            run.termination_reason = "user_stop"
            run.message = "Остановлено; backend освобождён"
        elif outcome is LifecycleOutcome.PAUSE:
            run.status = "paused"
            run.stop_requested = False
            run.paused = True
            run.termination_reason = "user_pause"
            run.message = "Пауза; backend освобождён"
        else:
            run.status = "completed"
            run.stop_requested = False
            run.paused = False
        self.storage.save_run(run)
        self.storage.log_action(
            "runtime",
            "run.lifecycle.finalized",
            {"run_id": run.id, "outcome": outcome.value, "status": run.status},
        )

    def _train(self, run_id: str) -> LifecycleOutcome:
        run = self.storage.get_run(run_id)
        if not run:
            raise RunError("Run исчез из storage")
        dataset = self.workspace.datasets.get(run.dataset)
        dataset_spec = dataset.manifest.get("streaming")
        if dataset_spec:
            x = y = None
        else:
            x, y = self.workspace.datasets.load_arrays(run.dataset)
        checkpoint = self.checkpoint_path(run.id)
        source = checkpoint if checkpoint.exists() else (self.checkpoint_path(run.base_run_id) if run.base_run_id else None)
        backend_name = run.effective_backend or run.backend
        total_memory = int(float(self.workspace.hardware.get("ram_gb") or 8) * 1024**3)
        gpu_memory = max(
            (int(float(item.get("vram_gb") or 0) * 1024) for item in self.workspace.hardware.get("nvidia_gpus", [])),
            default=0,
        )
        config_payload = {
            "dataset_hash": dataset.sha256,
            "backend": run.backend,
            "task_type": run.objective,
            "architecture": run.architecture,
            "model_format": run.model_format,
            "profile": run.profile,
            "seed": run.seed,
            "features": dataset.features,
            "streaming": bool(dataset_spec),
        }
        config_hash = hashlib.sha256(
            json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if source:
            self.checkpoints.verify(
                source,
                {
                    "dataset_hash": dataset.sha256,
                    "config_hash": config_hash,
                    "task_type": run.objective,
                    "backend": backend_name,
                },
            )
        config = BackendConfig(
            x=x,
            y=y,
            profile=run.profile,
            seed=run.seed,
            task_type=run.objective,
            resource_limits={
                "memory_bytes": max(256 * 1024**2, int(total_memory * run.ram / 100)),
                "cpu_percent": run.cpu,
                "gpu_memory_mb": int(gpu_memory * run.gpu / 100),
            },
            model_path=Path(run.model_path) if run.model_path else None,
            checkpoint_path=source,
            dataset_spec=dict(dataset_spec) if isinstance(dataset_spec, dict) else None,
            learning_rate=run.learning_rate,
            weight_decay=run.weight_decay,
            hidden_layers=tuple(run.hidden_layers),
            precision=run.precision,
            data_workers=run.data_workers,
            prefetch=run.prefetch,
        )
        worker = ProcessBackendWorker(
            backend_name,
            config,
            allow_numpy_fallback=True,
        )
        run.effective_backend = worker.effective_backend
        run.worker_protocol = worker.protocol_version
        run.worker_pid = worker.worker_pid
        run.resource_enforcement = worker.resource_enforcement
        run.environment_manifest = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "backend": worker.effective_backend,
            "backend_version": worker.metadata.get("backend_version", "unknown"),
            "worker_protocol": WORKER_PROTOCOL_VERSION,
            "dataset_hash": dataset.sha256,
            "config_hash": config_hash,
            "seed": run.seed,
            "architecture": run.architecture,
            "model_format": run.model_format,
        }
        self._save_worker_progress(run)
        return self._execute_worker(run_id, run, worker, checkpoint)

    def _execute_worker(
        self,
        run_id: str,
        run: RunInfo,
        worker: BackendWorker,
        checkpoint: Path,
    ) -> LifecycleOutcome:
        """Return an explicit outcome only after the backend session is released."""
        try:
            return self._run_worker_loop(run_id, run, worker, checkpoint)
        finally:
            worker.close()
            self._record_worker_events(run, worker)
            status = getattr(worker, "status", None)
            if status is not None and getattr(status, "forced_termination", False):
                run.message = f"{run.message}; worker завершён принудительно: {status.close_reason}"
                self.storage.log_action(
                    "runtime",
                    "worker.forced_termination",
                    {"run_id": run.id, "reason": status.close_reason},
                )
            run.worker_pid = None
            self._save_worker_progress(run)

    def _record_worker_events(self, run: RunInfo, worker: BackendWorker) -> None:
        drainer = getattr(worker, "drain_events", None)
        if not callable(drainer):
            return
        for event in drainer():
            self.storage.log_action(
                "worker",
                str(event.get("type", "event")),
                {"run_id": run.id, "payload": event.get("payload", {}), "created_at": event.get("created_at")},
            )

    def _save_worker_progress(self, run: RunInfo) -> None:
        """Save metrics without overwriting a concurrent control-plane transition."""
        with self.lock:
            current = self.storage.get_run(run.id)
            if current:
                transition_message = current.message if current.status in {"pausing", "stopping"} else None
                run.status = current.status
                run.paused = current.paused
                run.stop_requested = current.stop_requested
                if transition_message is not None:
                    run.message = transition_message
            self.storage.save_run(run)

    def _run_worker_loop(
        self,
        run_id: str,
        run: RunInfo,
        worker: BackendWorker,
        checkpoint: Path,
    ) -> LifecycleOutcome:
        recent_scores: list[float] = []
        plateau = 0
        target_hits = 0
        degradation_streak = 0
        last_progress_save = 0.0
        evaluation_interval = run.evaluation_interval or max(20, min(500, max(1, run.max_steps // 20)))
        while True:
            with self.lock:
                control = dict(self.controls.get(run_id, {"pause": False, "stop": False}))
            if control["stop"]:
                run.message = "Остановка подтверждена; backend освобождается"
                self._save_worker_progress(run)
                return LifecycleOutcome.STOP
            if control["pause"]:
                run.message = "Пауза подтверждена; backend освобождается"
                self._save_worker_progress(run)
                return LifecycleOutcome.PAUSE
            steps_to_evaluation = evaluation_interval - (run.step % evaluation_interval)
            steps_to_finish = max(1, run.max_steps - run.step)
            fast_linear_backend = (run.effective_backend or run.backend) in {"native", "numpy"}
            maximum_chunk = 100 if fast_linear_backend else 1
            responsiveness_cap = max(1, min(maximum_chunk, 65_536 // max(1, run.batch_size)))
            chunk_steps = min(steps_to_evaluation, steps_to_finish, responsiveness_cap)
            try:
                train_steps = getattr(worker, "train_steps", None)
                if chunk_steps > 1 and callable(train_steps):
                    step = train_steps(run.batch_size, chunk_steps)
                    executed_steps = chunk_steps
                else:
                    step = worker.train_step(run.batch_size)
                    executed_steps = 1
                self._record_worker_events(run, worker)
            except Exception as error:
                is_oom = type(error).__name__ == "ResourceExhaustedError" or "out of memory" in str(error).lower()
                if is_oom and run.batch_mode == "auto" and run.batch_size > run.batch_min:
                    previous = run.batch_size
                    run.batch_size = max(run.batch_min, run.batch_size // 2)
                    run.batch_adjustments += 1
                    run.message = f"OOM предотвращён: batch снижен {previous} → {run.batch_size}"
                    self._save_worker_progress(run)
                    continue
                raise
            run.step += executed_steps
            run.samples_seen += step.samples
            run.loss = step.loss
            if run.step % evaluation_interval != 0 and run.step < run.max_steps:
                now = time.monotonic()
                if last_progress_save == 0.0 or now - last_progress_save >= .5:
                    run.message = f"Обучение: шаг {run.step}; loss {run.loss:.4f}; batch {run.batch_size}"
                    self._save_worker_progress(run)
                    last_progress_save = now
                continue
            evaluation = worker.evaluate()
            self._record_worker_events(run, worker)
            score = evaluation.score
            run.eval_count += 1
            run.metrics = {
                "step": run.step,
                "train": {"loss": run.loss},
                "validation": evaluation.metrics,
                "created_at": time.time(),
            }
            recent_scores.append(score)
            recent_scores = recent_scores[-4:]
            checkpoint_metadata = {
                "run_id": run.id,
                "score": score,
                "step": run.step,
                "schema_version": 3,
                "dataset_hash": run.environment_manifest.get("dataset_hash", ""),
                "config_hash": run.environment_manifest.get("config_hash", ""),
                "backend": run.effective_backend or run.backend,
                "task_type": run.objective,
                "architecture": run.architecture,
                "model_format": run.model_format,
                "worker_protocol": run.worker_protocol,
            }
            last_checkpoint = self.checkpoints.last_path(run.id)
            worker.save_checkpoint(last_checkpoint, checkpoint_metadata)
            self._record_worker_events(run, worker)
            last_info = (
                self.checkpoints.record(last_checkpoint, checkpoint_metadata)
                if last_checkpoint.is_file()
                else None
            )
            previous_best = run.best_score
            if run.eval_count > 1 and score < previous_best - run.degradation_margin:
                degradation_streak += 1
            else:
                degradation_streak = 0
            if run.eval_count == 1 or score > run.best_score + run.min_improvement:
                run.best_score = score
                run.best_step = run.step
                plateau = 0
                if last_info is not None:
                    best_info = self.checkpoints.promote(
                        last_checkpoint,
                        checkpoint,
                        {**checkpoint_metadata, "best_score": run.best_score, "kind": "best"},
                    )
                    run.checkpoint = {
                        "best": {"filename": checkpoint.name, "sha256": best_info["sha256"], "step": run.step},
                        "last": {
                            "filename": last_checkpoint.name,
                            "sha256": last_info["sha256"],
                            "step": run.step,
                        },
                    }
            else:
                plateau += 1
                if last_info is not None:
                    run.checkpoint["last"] = {
                        "filename": last_checkpoint.name,
                        "sha256": last_info["sha256"],
                        "step": run.step,
                    }
            run.metrics["health"] = {
                "status": "degrading" if degradation_streak else "stable",
                "degradation_streak": degradation_streak,
                "gap_from_best": max(0.0, run.best_score - score),
                "best_score": run.best_score,
                "policy": {
                    "evaluation_interval": evaluation_interval,
                    "min_improvement": run.min_improvement,
                    "degradation_margin": run.degradation_margin,
                    "degradation_patience": run.degradation_patience,
                    "early_stopping_patience": run.early_stopping_patience,
                    "target_patience": run.target_patience,
                },
            }
            target_hits = target_hits + 1 if score >= run.target_score else 0
            train_records = int(getattr(worker, "train_records", 0) or 0)
            if not train_records and hasattr(worker, "session"):
                train_records = len(worker.session.train_x)
            note = self._adapt_batch(run, recent_scores, train_records)
            run.message = note or f"Validation #{run.eval_count}: score {score:.3f}; batch {run.batch_size}"
            if degradation_streak >= run.degradation_patience:
                run.message = "Model Guard: качество ухудшается; восстановлен лучший checkpoint"
                run.termination_reason = "model_degradation"
                rejected_checkpoint = dict(run.checkpoint.get("last") or {})
                self.checkpoints.discard_last(run.id)
                run.checkpoint.pop("last", None)
                run.metrics["health"] = {
                    **run.metrics["health"],
                    "status": "best_checkpoint_restored",
                    "delivered_score": run.best_score,
                    "rejected_score": score,
                    "rejected_checkpoint": {
                        **rejected_checkpoint,
                        "discarded": True,
                    },
                }
                self._save_worker_progress(run)
                return self._complete_with_test(run, worker, checkpoint)
            if target_hits >= run.target_patience:
                run.message = f"Цель {run.target_score:.3f} достигнута устойчиво; сохранён лучший checkpoint"
                run.termination_reason = "target_reached"
                self._save_worker_progress(run)
                return self._complete_with_test(run, worker, checkpoint)
            if plateau > run.early_stopping_patience:
                run.message = "Smart Stop: улучшение закончилось; сохранён лучший checkpoint"
                run.termination_reason = "early_stopping"
                self._save_worker_progress(run)
                return self._complete_with_test(run, worker, checkpoint)
            if run.step >= run.max_steps:
                run.step = run.max_steps
                run.message = "Аварийный лимит шагов достигнут; сохранён лучший checkpoint"
                run.termination_reason = "max_steps"
                self._save_worker_progress(run)
                return self._complete_with_test(run, worker, checkpoint)
            self._save_worker_progress(run)

    def _complete_with_test(
        self,
        run: RunInfo,
        worker: BackendWorker,
        checkpoint: Path,
    ) -> LifecycleOutcome:
        restorer = getattr(worker, "restore_checkpoint", None)
        if checkpoint.exists() and callable(restorer):
            restorer(checkpoint)
            self._record_worker_events(run, worker)
        evaluator = getattr(worker, "evaluate_test", None)
        if not callable(evaluator):
            run.metrics = {**run.metrics, "test_unavailable": "legacy_backend_worker_contract"}
            self._save_worker_progress(run)
            return LifecycleOutcome.COMPLETE
        test_result = evaluator()
        self._record_worker_events(run, worker)
        run.metrics = {
            **run.metrics,
            "test": test_result.metrics,
            "test_score": test_result.score,
            "tested_checkpoint": run.checkpoint.get("best", {}).get("sha256"),
        }
        self._save_worker_progress(run)
        return LifecycleOutcome.COMPLETE

    @staticmethod
    def _adapt_batch(run: RunInfo, scores: list[float], train_records: int) -> str | None:
        if run.batch_mode != "auto":
            return None
        current = run.batch_size
        if len(scores) >= 3 and scores[-1] < max(scores[-3:-1]) - .04 and current > run.batch_min:
            run.batch_size = max(run.batch_min, current // 2)
            run.batch_adjustments += 1
            return f"Batch снижен {current} → {run.batch_size}: validation стала нестабильной"
        stable = len(scores) >= 2 and abs(scores[-1] - scores[-2]) <= .02
        if run.eval_count in {2, 4} and stable and current < run.batch_max and train_records >= current * 8:
            run.batch_size = min(run.batch_max, current * 2)
            run.batch_adjustments += 1
            return f"Batch увеличен {current} → {run.batch_size}: качество устойчиво"
        return None

    def rebalance(self) -> dict[str, Any]:
        active = [run for run in self.storage.list_runs() if run.status == "training"]
        if active and self.workspace.hardware["gpu_available"]:
            weights = [{"high": 1.5, "normal": 1.0, "low": .65}.get(run.priority, 1.0) for run in active]
            total = sum(weights)
            for run, weight in zip(active, weights, strict=False):
                run.gpu = max(5, round(self.GPU_CAP * weight / total))
                self.storage.save_run(run)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        runs = self.storage.list_runs()
        datasets = self.workspace.datasets.list()
        recommended = recommend_profile(self.workspace.hardware)
        profiles = [
            {"id": key, **value, "recommended": key == recommended}
            for key, value in PROFILE_PRESETS.items()
            if key != "eco"
        ]
        return {
            "runs": [run.to_dict() for run in runs],
            "usage": self.usage(),
            "capacity": {
                "gpu": self.GPU_CAP if self.workspace.hardware["gpu_available"] else 0,
                "ram": self.RAM_CAP,
                "cpu": self.CPU_CAP,
            },
            "datasets": [item.to_dict() for item in datasets],
            "hardware": self.workspace.hardware,
            "profiles": profiles,
            "recommended_profile": recommended,
            "model_formats": self.workspace.capabilities["model_formats"],
            "format_router_minimum": self.workspace.datasets.router.route_count,
            "capabilities": self.workspace.capabilities,
            "api_version": "v1",
            "resource_enforcement": self.workspace.hardware.get("resource_enforcement", {}),
        }
