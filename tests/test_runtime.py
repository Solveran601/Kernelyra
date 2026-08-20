from __future__ import annotations

import time
import unittest

from kernelyra import RunConfig, Workspace
from tests.helpers import isolated_workspace


class RuntimeTests(unittest.TestCase):
    def test_numpy_run_executes_only_after_explicit_start(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            handle = workspace.create_run(RunConfig(dataset="demo", backend="numpy", name="runtime-check", target_metric=.5, max_steps=100, seed=7))
            self.assertEqual(handle.info.status, "draft")
            handle.start()
            deadline = time.time() + 15
            while handle.info.status not in {"completed", "error_recoverable"} and time.time() < deadline:
                time.sleep(.1)
            result = handle.info
            workspace.close()
            self.assertEqual(result.status, "completed", result.message)
            self.assertGreater(result.samples_seen, 0)
            self.assertGreater(result.eval_count, 0)
            self.assertTrue((workspace.state_dir / "checkpoints" / f"{result.id}.npz").exists())


if __name__ == "__main__":
    unittest.main()
