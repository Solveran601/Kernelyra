from __future__ import annotations

import time
import unittest

from fastapi.testclient import TestClient

from kernelyra.server import create_app
from tests.helpers import isolated_workspace


class StateMachineTests(unittest.TestCase):
    def test_duplicate_start_is_idempotent_and_completed_start_conflicts(self) -> None:
        with isolated_workspace() as temporary:
            project = temporary / "project"
            app = create_app(project)
            secret = (project / ".kernelyra" / "daemon.secret").read_text(encoding="ascii")
            headers = {"X-Kernelyra-User-Secret": secret}
            with TestClient(app) as client:
                created = client.post(
                    "/api/v1/runs",
                    headers=headers,
                    json={"dataset": "demo", "backend": "numpy", "start": False, "max_steps": 10000},
                ).json()
                first = client.post(f"/api/v1/runs/{created['id']}/command", headers=headers, json={"command": "start"})
                second = client.post(f"/api/v1/runs/{created['id']}/command", headers=headers, json={"command": "start"})
                self.assertEqual(first.status_code, 200)
                self.assertEqual(second.status_code, 200)
                self.assertEqual(first.json()["status"], "queued")
                self.assertEqual(second.json()["status"], "queued")

                paused = client.post(f"/api/v1/runs/{created['id']}/command", headers=headers, json={"command": "pause"})
                self.assertEqual(paused.json()["status"], "paused")
                resumed = client.post(f"/api/v1/runs/{created['id']}/command", headers=headers, json={"command": "resume"})
                self.assertEqual(resumed.json()["status"], "queued")
                stopped = client.post(f"/api/v1/runs/{created['id']}/command", headers=headers, json={"command": "stop"})
                self.assertIn(stopped.json()["status"], {"stopping", "stopped"})
                deadline = time.monotonic() + 10
                while app.state.workspace.storage.get_run(created["id"]).status != "stopped":
                    if time.monotonic() >= deadline:
                        self.fail("run did not finish stopping")
                    time.sleep(.05)

                run = app.state.workspace.storage.get_run(created["id"])
                run.status = "completed"
                app.state.workspace.storage.save_run(run)
                conflict = client.post(f"/api/v1/runs/{created['id']}/command", headers=headers, json={"command": "start"})
                self.assertEqual(conflict.status_code, 409)
                self.assertEqual(conflict.json()["type"], "RunStateError")


if __name__ == "__main__":
    unittest.main()
