from __future__ import annotations

import os
import unittest

from kernelyra.backends import WORKER_PROTOCOL_VERSION, BackendConfig, ProcessBackendWorker
from kernelyra.errors import WorkerCrashedError
from kernelyra.workspace import Workspace
from tests.helpers import isolated_workspace


class ProcessBackendWorkerTests(unittest.TestCase):
    def test_spawned_worker_protocol_metrics_checkpoint_and_close(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            x, y = workspace.datasets.load_arrays("demo")
            worker = ProcessBackendWorker(
                "numpy",
                BackendConfig(
                    x=x,
                    y=y,
                    profile="low-memory",
                    seed=7,
                    task_type="binary_classification",
                    resource_limits={"memory_bytes": 1024**3, "cpu_percent": 50},
                ),
            )
            try:
                self.assertEqual(worker.protocol_version, WORKER_PROTOCOL_VERSION)
                self.assertNotEqual(worker.worker_pid, os.getpid())
                step = worker.train_step(16)
                self.assertGreater(step.samples, 0)
                chunk = worker.train_steps(16, 3)
                self.assertEqual(chunk.samples, 48)
                evaluation = worker.evaluate()
                self.assertIn("accuracy", evaluation.metrics)
                checkpoint = temporary / "checkpoint.npz"
                worker.save_checkpoint(checkpoint, {"schema_version": 3})
                self.assertTrue(checkpoint.exists())
            finally:
                worker.close()
                workspace.close()

    def test_worker_crash_is_contained_and_a_new_worker_still_runs(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            x, y = workspace.datasets.load_arrays("demo")
            config = BackendConfig(x=x, y=y, profile="low-memory", seed=9)
            crashed = ProcessBackendWorker("numpy", config)
            crashed._process.terminate()
            crashed._process.join(timeout=5)
            with self.assertRaises(WorkerCrashedError):
                crashed.train_step(8)
            crashed.close()
            replacement = ProcessBackendWorker("numpy", config)
            try:
                self.assertGreater(replacement.train_step(8).samples, 0)
            finally:
                replacement.close()
                workspace.close()


if __name__ == "__main__":
    unittest.main()
