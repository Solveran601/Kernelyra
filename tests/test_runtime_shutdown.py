from __future__ import annotations

import threading
import unittest

from kernelyra import Workspace
from kernelyra.errors import ConfigurationError
from tests.helpers import isolated_workspace


class RuntimeShutdownTests(unittest.TestCase):
    def test_scheduler_lock_is_retained_while_worker_is_alive(self) -> None:
        with isolated_workspace() as temporary:
            root = temporary / "project"
            first = Workspace.open(root)
            first.runtime.start()
            release = threading.Event()
            worker = threading.Thread(target=lambda: release.wait(5), name="slow-backend-worker", daemon=True)
            worker.start()
            first.runtime.threads["slow"] = worker

            self.assertFalse(first.runtime.close(timeout=.01))
            self.assertIsNotNone(first.runtime._scheduler_lock_handle)

            second = Workspace.open(root)
            with self.assertRaises(ConfigurationError):
                second.runtime.start()

            release.set()
            worker.join(timeout=2)
            self.assertTrue(first.runtime.close(timeout=1))
            self.assertIsNone(first.runtime._scheduler_lock_handle)
            second.runtime.start()
            self.assertTrue(second.close())


if __name__ == "__main__":
    unittest.main()
