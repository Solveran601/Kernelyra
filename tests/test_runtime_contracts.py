from __future__ import annotations

import math
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from kernelyra import RunConfig, Workspace
from kernelyra.backends.base import EvaluationResult, StepResult
from kernelyra.errors import RunError, RunStateError
from kernelyra.runtime import LifecycleOutcome, TrainingRuntime
from tests.helpers import isolated_workspace


class ResourceExhaustedError(RuntimeError):
    pass


class ControlledWorker:
    effective_backend = "test"
    protocol_version = "test/1"
    worker_pid: int | None = None
    resource_enforcement: dict[str, object] = {}
    train_records = 32

    def __init__(self, runtime: TrainingRuntime, run_id: str, *, oom: bool = False, forced: bool = False):
        self.runtime = runtime
        self.run_id = run_id
        self.oom = oom
        self.closed = False
        self.status = SimpleNamespace(forced_termination=forced, close_reason="test-forced")

    def train_step(self, batch_size: int):
        if self.oom:
            self.runtime.controls[self.run_id]["pause"] = True
            raise ResourceExhaustedError("synthetic out of memory")
        raise AssertionError("train_step should not be reached")

    def close(self) -> None:
        self.closed = True

    def drain_events(self) -> list[dict[str, object]]:
        return [{"type": "test.worker", "payload": {"ok": True}}]


class DegradingWorker:
    train_records = 1000

    def __init__(self):
        self.scores = iter((.90, .80, .79, .78))
        self.restored = None
        self.tested = False

    @staticmethod
    def train_steps(batch_size: int, steps: int) -> StepResult:
        return StepResult(loss=.25, samples=batch_size * steps)

    def evaluate(self) -> EvaluationResult:
        score = next(self.scores)
        return EvaluationResult(score, {"accuracy": score, "loss": 1 - score})

    def evaluate_test(self) -> EvaluationResult:
        self.tested = True
        return EvaluationResult(.88, {"accuracy": .88, "loss": .12})

    @staticmethod
    def save_checkpoint(path, metadata) -> None:
        path.write_bytes(f"checkpoint:{metadata['step']}".encode("ascii"))

    def restore_checkpoint(self, path) -> None:
        self.restored = path

    @staticmethod
    def drain_events() -> list[dict[str, object]]:
        return []


class ObservableSlowWorker:
    train_records = 1000

    def __init__(self, runtime: TrainingRuntime, run_id: str):
        self.runtime = runtime
        self.run_id = run_id
        self.calls = 0

    def train_step(self, batch_size: int) -> StepResult:
        self.calls += 1
        if self.calls == 2:
            self.runtime.controls[self.run_id]["pause"] = True
        return StepResult(loss=.25, samples=batch_size)

    @staticmethod
    def drain_events() -> list[dict[str, object]]:
        return []


class InvalidMetricWorker:
    train_records = 1000

    @staticmethod
    def train_steps(batch_size: int, steps: int) -> StepResult:
        return StepResult(loss=.25, samples=batch_size * steps)

    @staticmethod
    def evaluate() -> EvaluationResult:
        return EvaluationResult(math.nan, {"accuracy": math.nan})

    @staticmethod
    def drain_events() -> list[dict[str, object]]:
        return []


class RuntimeContractTests(unittest.TestCase):
    def test_restart_recovery_double_lock_and_snapshot(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            run = workspace.create_run(RunConfig(dataset="demo", backend="numpy")).info
            run.status = "training"
            workspace.storage.save_run(run)

            runtime = workspace.runtime
            runtime.start()
            runtime._acquire_scheduler_lock()
            recovered = workspace.storage.get_run(run.id)
            self.assertEqual(recovered.status, "error_recoverable")
            self.assertEqual(recovered.termination_reason, "daemon_restart")
            snapshot = runtime.snapshot()
            self.assertEqual(snapshot["api_version"], "v1")
            self.assertNotIn("eco", {item["id"] for item in snapshot["profiles"]})
            self.assertTrue(workspace.close())

    def test_close_marks_live_training_run_as_pausing(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            runtime = workspace.runtime
            run = workspace.create_run(RunConfig(dataset="demo", backend="numpy")).info
            run.status = "training"
            workspace.storage.save_run(run)
            release = threading.Event()
            thread = threading.Thread(target=release.wait, args=(2,), daemon=True)
            runtime.threads[run.id] = thread
            thread.start()
            self.assertFalse(runtime.close(timeout=.01))
            self.assertEqual(workspace.storage.get_run(run.id).status, "pausing")
            release.set()
            thread.join(timeout=1)
            self.assertTrue(runtime.close(timeout=1))

    def test_command_rejections_and_idempotent_terminal_commands(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            runtime = workspace.runtime
            run = workspace.create_run(RunConfig(dataset="demo", backend="numpy")).info
            with self.assertRaises(RunError):
                runtime.command("missing-run", "start")
            with self.assertRaises(RunStateError):
                runtime.command(run.id, "resume")
            with self.assertRaises(RunStateError):
                runtime.command(run.id, "pause")
            with self.assertRaises(RunError):
                runtime.command(run.id, "unknown")

            run.status = "paused"
            workspace.storage.save_run(run)
            self.assertEqual(runtime.command(run.id, "pause").status, "paused")
            run.status = "stopped"
            workspace.storage.save_run(run)
            self.assertEqual(runtime.command(run.id, "stop").status, "stopped")
            run.status = "completed"
            workspace.storage.save_run(run)
            with self.assertRaises(RunStateError):
                runtime.command(run.id, "stop")
            workspace.close()

    def test_guarded_failure_missing_run_oom_and_forced_close(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            runtime = workspace.runtime
            run = workspace.create_run(RunConfig(dataset="demo", backend="numpy")).info
            run.status = "training"
            workspace.storage.save_run(run)
            runtime.controls[run.id] = {"pause": False, "stop": False}
            with patch.object(runtime, "_train", side_effect=RuntimeError("guarded crash")):
                runtime._train_guarded(run.id)
            self.assertEqual(workspace.storage.get_run(run.id).status, "error_recoverable")
            with self.assertRaises(RunError):
                runtime._train("missing-run")

            oom_run = workspace.create_run(RunConfig(dataset="demo", backend="numpy")).info
            oom_run.batch_size = max(oom_run.batch_min * 2, 16)
            runtime.controls[oom_run.id] = {"pause": False, "stop": False}
            worker = ControlledWorker(runtime, oom_run.id, oom=True)
            outcome = runtime._run_worker_loop(
                oom_run.id,
                oom_run,
                worker,
                runtime.checkpoint_path(oom_run.id),
            )
            self.assertIs(outcome, LifecycleOutcome.PAUSE)
            self.assertEqual(oom_run.batch_adjustments, 1)

            forced_run = workspace.create_run(RunConfig(dataset="demo", backend="numpy")).info
            runtime.controls[forced_run.id] = {"pause": False, "stop": True}
            forced = ControlledWorker(runtime, forced_run.id, forced=True)
            outcome = runtime._execute_worker(
                forced_run.id,
                forced_run,
                forced,
                runtime.checkpoint_path(forced_run.id),
            )
            self.assertIs(outcome, LifecycleOutcome.STOP)
            self.assertTrue(forced.closed)
            actions = workspace.storage.recent_actions(20)
            self.assertTrue(any(item["action"] == "worker.forced_termination" for item in actions))
            workspace.close()

    def test_adaptive_batch_and_gpu_rebalance_contracts(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            runtime = workspace.runtime
            run = workspace.create_run(RunConfig(dataset="demo", backend="numpy")).info
            run.batch_mode = "manual"
            self.assertIsNone(runtime._adapt_batch(run, [.8, .7], 1000))
            run.batch_mode = "auto"
            original = run.batch_size
            self.assertIsNotNone(runtime._adapt_batch(run, [.95, .94, .80], 1000))
            self.assertLess(run.batch_size, original)
            run.eval_count = 2
            run.batch_size = run.batch_min
            self.assertIsNotNone(runtime._adapt_batch(run, [.80, .81], run.batch_size * 8))

            second = workspace.create_run(RunConfig(dataset="demo", backend="numpy")).info
            run.status = second.status = "training"
            run.priority = "high"
            second.priority = "low"
            workspace.storage.save_run(run)
            workspace.storage.save_run(second)
            workspace.hardware["gpu_available"] = True
            snapshot = runtime.rebalance()
            self.assertGreater(workspace.storage.get_run(run.id).gpu, workspace.storage.get_run(second.id).gpu)
            self.assertEqual(snapshot["capacity"]["gpu"], runtime.GPU_CAP)
            workspace.close()

    def test_model_guard_stops_degradation_and_tests_the_best_checkpoint(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            runtime = workspace.runtime
            run = workspace.create_run(
                RunConfig(dataset="demo", backend="native", max_steps=100)
            ).info
            run.effective_backend = "native"
            run.batch_mode = "manual"
            runtime.controls[run.id] = {"pause": False, "stop": False}
            worker = DegradingWorker()
            best = runtime.checkpoint_path(run.id)
            outcome = runtime._run_worker_loop(run.id, run, worker, best)
            self.assertIs(outcome, LifecycleOutcome.COMPLETE)
            self.assertEqual(run.termination_reason, "model_degradation")
            self.assertEqual(run.step, 80)
            self.assertEqual(worker.restored, best)
            self.assertTrue(worker.tested)
            self.assertEqual(run.metrics["health"]["degradation_streak"], 3)
            self.assertEqual(run.metrics["health"]["status"], "best_checkpoint_restored")
            self.assertEqual(run.metrics["health"]["delivered_score"], run.best_score)
            self.assertTrue(run.metrics["health"]["rejected_checkpoint"]["discarded"])
            self.assertFalse(runtime.checkpoints.last_path(run.id).exists())
            self.assertNotIn("last", run.checkpoint)
            self.assertEqual(run.metrics["test_score"], .88)
            self.assertTrue(any(item["event"] == "evaluation" for item in run.metrics["trace"]))
            self.assertEqual(run.metrics["quality_gate"]["status"], "regressing")
            workspace.close()

    def test_finetune_baseline_is_preserved_when_updates_degrade(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            runtime = workspace.runtime
            run = workspace.create_run(
                RunConfig(dataset="demo", backend="native", max_steps=100, evaluation_interval=10)
            ).info
            run.effective_backend = "native"
            run.model_path = "incoming-model.npz"
            run.batch_mode = "manual"
            runtime.controls[run.id] = {"pause": False, "stop": False}
            worker = DegradingWorker()
            best = runtime.checkpoint_path(run.id)
            runtime._capture_finetune_baseline(
                run,
                worker,
                best,
                {
                    "run_id": run.id,
                    "dataset_hash": "dataset",
                    "config_hash": "config",
                    "backend": "native",
                    "task_type": run.objective,
                    "architecture": run.architecture,
                    "model_format": run.model_format,
                    "worker_protocol": "test/1",
                },
            )
            outcome = runtime._run_worker_loop(run.id, run, worker, best)
            self.assertIs(outcome, LifecycleOutcome.COMPLETE)
            self.assertEqual(run.termination_reason, "model_degradation")
            self.assertEqual(run.best_score, .90)
            self.assertEqual(run.best_step, 0)
            self.assertEqual(run.metrics["health"]["delivered_score"], .90)
            self.assertEqual(run.environment_manifest["fine_tune_baseline"]["score"], .90)
            self.assertTrue(any(item["event"] == "fine_tune_baseline" for item in run.metrics["trace"]))
            self.assertEqual(worker.restored, best)
            actions = workspace.storage.recent_actions(20)
            self.assertTrue(any(item["action"] == "finetune.baseline_preserved" for item in actions))
            workspace.close()

    def test_model_guard_manual_sensitivity_is_applied_and_reported(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            runtime = workspace.runtime
            run = workspace.create_run(
                RunConfig(
                    dataset="demo",
                    backend="native",
                    max_steps=100,
                    evaluation_interval=10,
                    degradation_margin=.02,
                    degradation_patience=1,
                    min_improvement=.001,
                    early_stopping_patience=4,
                    target_patience=2,
                )
            ).info
            run.effective_backend = "native"
            run.batch_mode = "manual"
            runtime.controls[run.id] = {"pause": False, "stop": False}
            worker = DegradingWorker()
            best = runtime.checkpoint_path(run.id)
            outcome = runtime._run_worker_loop(run.id, run, worker, best)
            self.assertIs(outcome, LifecycleOutcome.COMPLETE)
            self.assertEqual(run.termination_reason, "model_degradation")
            self.assertEqual(run.step, 20)
            policy = run.metrics["health"]["policy"]
            self.assertEqual(policy["evaluation_interval"], 10)
            self.assertEqual(policy["degradation_patience"], 1)
            self.assertEqual(policy["degradation_margin"], .02)
            workspace.close()

    def test_quality_gate_rejects_non_finite_metrics_before_checkpoint_write(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            runtime = workspace.runtime
            run = workspace.create_run(
                RunConfig(dataset="demo", backend="native", max_steps=100, evaluation_interval=10)
            ).info
            run.effective_backend = "native"
            run.batch_mode = "manual"
            runtime.controls[run.id] = {"pause": False, "stop": False}
            with self.assertRaisesRegex(FloatingPointError, "Quality Gate"):
                runtime._run_worker_loop(
                    run.id, run, InvalidMetricWorker(), runtime.checkpoint_path(run.id)
                )
            self.assertEqual(run.termination_reason, "quality_gate_invalid")
            self.assertFalse(run.metrics["quality_gate"]["finite"])
            self.assertNotIn("validation", run.metrics)
            self.assertFalse(runtime.checkpoints.last_path(run.id).exists())
            workspace.close()

    def test_slow_worker_progress_is_published_before_validation(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            runtime = workspace.runtime
            run = workspace.create_run(
                RunConfig(dataset="demo", backend="numpy", max_steps=10_000)
            ).info
            run.status = "training"
            run.backend = "third-party"
            run.effective_backend = "third-party"
            workspace.storage.save_run(run)
            runtime.controls[run.id] = {"pause": False, "stop": False}
            worker = ObservableSlowWorker(runtime, run.id)

            outcome = runtime._run_worker_loop(
                run.id,
                run,
                worker,
                runtime.checkpoint_path(run.id),
            )

            self.assertIs(outcome, LifecycleOutcome.PAUSE)
            saved = workspace.storage.get_run(run.id)
            self.assertIsNotNone(saved)
            self.assertGreaterEqual(saved.step, 1)
            self.assertEqual(saved.samples_seen, saved.step * run.batch_size)
            workspace.close()


if __name__ == "__main__":
    unittest.main()
