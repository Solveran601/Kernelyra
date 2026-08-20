from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from kernelyra import RunConfig, Workspace
from kernelyra.backends import EvaluationResult, StepResult
from kernelyra.runtime import LifecycleOutcome
from tests.helpers import isolated_workspace


class FakeWorker:
    def __init__(self, *, fail: bool = False):
        self.session = type("Session", (), {"train_x": np.zeros((16, 1))})()
        self.fail = fail
        self.closed = False

    def train_step(self, batch_size: int) -> StepResult:
        if self.fail:
            raise RuntimeError("synthetic backend failure")
        return StepResult(loss=.1, samples=batch_size)

    def evaluate(self) -> EvaluationResult:
        return EvaluationResult(score=.9)

    def save_checkpoint(self, path, metadata) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class BlockingCloseWorker(FakeWorker):
    def __init__(self):
        super().__init__()
        self.step_started = threading.Event()
        self.release_step = threading.Event()
        self.close_started = threading.Event()
        self.release_close = threading.Event()

    def train_step(self, batch_size: int) -> StepResult:
        self.step_started.set()
        if not self.release_step.wait(5):
            raise TimeoutError("test did not release train_step")
        return StepResult(loss=.1, samples=batch_size)

    def close(self) -> None:
        self.close_started.set()
        if not self.release_close.wait(5):
            raise TimeoutError("test did not release close")
        self.closed = True


class RuntimeResourceTests(unittest.TestCase):
    def test_transitional_worker_keeps_resource_slot_until_thread_exits(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            runtime = workspace.runtime
            runtime.start()
            old = workspace.create_run(RunConfig(dataset="demo", backend="numpy", cpu=60, ram=60)).info
            pending = workspace.create_run(RunConfig(dataset="demo", backend="numpy", cpu=40, ram=30)).info
            old.status = "training"
            pending.status = "queued"
            workspace.storage.save_run(old)
            workspace.storage.save_run(pending)

            release = threading.Event()
            worker_thread = threading.Thread(target=release.wait, name="synthetic-live-worker", daemon=True)
            runtime.controls[old.id] = {"pause": False, "stop": False}
            runtime.threads[old.id] = worker_thread
            worker_thread.start()

            paused = runtime.command(old.id, "pause")
            self.assertEqual(paused.status, "pausing")
            self.assertEqual(runtime.usage()["ram"], 60)
            time.sleep(.6)
            self.assertEqual(workspace.storage.get_run(pending.id).status, "queued")
            self.assertNotIn(pending.id, runtime.threads)

            stopping = runtime.command(old.id, "stop")
            self.assertEqual(stopping.status, "stopping")
            runtime.command(pending.id, "stop")
            release.set()
            worker_thread.join(timeout=2)
            runtime.threads.pop(old.id, None)
            runtime.controls.pop(old.id, None)
            workspace.close()

    def test_worker_close_runs_for_completion_pause_stop_and_error(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            runtime = workspace.runtime
            for outcome in ("completed", "paused", "stopped", "error"):
                with self.subTest(outcome=outcome):
                    run = workspace.create_run(RunConfig(dataset="demo", backend="numpy")).info
                    run.max_steps = 1
                    workspace.storage.save_run(run)
                    runtime.controls[run.id] = {"pause": outcome == "paused", "stop": outcome == "stopped"}
                    worker = FakeWorker(fail=outcome == "error")
                    failure = None
                    if outcome == "error":
                        try:
                            runtime._execute_worker(run.id, run, worker, runtime.checkpoint_path(run.id))
                        except RuntimeError as error:
                            failure = error
                        else:
                            self.fail("synthetic backend failure was not raised")
                        lifecycle = LifecycleOutcome.ERROR
                    else:
                        lifecycle = runtime._execute_worker(run.id, run, worker, runtime.checkpoint_path(run.id))
                    expected = {
                        "completed": LifecycleOutcome.COMPLETE,
                        "paused": LifecycleOutcome.PAUSE,
                        "stopped": LifecycleOutcome.STOP,
                        "error": LifecycleOutcome.ERROR,
                    }[outcome]
                    self.assertIs(lifecycle, expected)
                    self.assertTrue(worker.closed)
                    with runtime.lock:
                        runtime._finalize_lifecycle_locked(run, lifecycle, failure)
                    expected_status = "error_recoverable" if outcome == "error" else outcome
                    self.assertEqual(workspace.storage.get_run(run.id).status, expected_status)
                    runtime.controls.pop(run.id, None)
            workspace.close()

    def test_stop_during_blocking_close_finishes_as_stopped(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            runtime = workspace.runtime
            run = workspace.create_run(RunConfig(dataset="demo", backend="numpy", max_steps=100)).info
            run.status = "training"
            workspace.storage.save_run(run)
            runtime.controls[run.id] = {"pause": False, "stop": False}
            worker = BlockingCloseWorker()

            original_train = runtime._train
            runtime._train = lambda run_id: runtime._execute_worker(
                run_id, run, worker, runtime.checkpoint_path(run_id)
            )
            thread = threading.Thread(
                target=runtime._train_guarded,
                args=(run.id,),
                name=f"blocking-close-{run.id}",
                daemon=True,
            )
            runtime.threads[run.id] = thread
            try:
                thread.start()
                self.assertTrue(worker.step_started.wait(2))
                self.assertEqual(runtime.command(run.id, "pause").status, "pausing")
                worker.release_step.set()
                self.assertTrue(worker.close_started.wait(2))
                self.assertTrue(thread.is_alive())
                during_close = workspace.storage.get_run(run.id)
                self.assertEqual(during_close.status, "pausing")
                self.assertTrue(during_close.paused)

                self.assertEqual(runtime.command(run.id, "stop").status, "stopping")
                worker.release_close.set()
                thread.join(timeout=3)
                self.assertFalse(thread.is_alive())
                self.assertTrue(worker.closed)
                self.assertEqual(workspace.storage.get_run(run.id).status, "stopped")
                self.assertNotIn(run.id, runtime.threads)
            finally:
                worker.release_step.set()
                worker.release_close.set()
                thread.join(timeout=1)
                runtime._train = original_train
                workspace.close()


if __name__ == "__main__":
    unittest.main()
