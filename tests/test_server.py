from __future__ import annotations

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from kernelyra.server import create_app
from tests.helpers import isolated_workspace

ROOT = Path(__file__).resolve().parents[1]


class ServerTests(unittest.TestCase):
    def test_versioned_api_and_batch_guard(self) -> None:
        with isolated_workspace() as temporary:
            project = temporary / "project"
            app = create_app(project, ROOT)
            secret = (project / ".kernelyra" / "daemon.secret").read_text(encoding="ascii")
            headers = {"X-Kernelyra-User-Secret": secret}
            with TestClient(app) as client:
                health = client.get("/api/v1/health")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(set(health.json()), {"ok", "version", "protocol"})
                private_health = client.get("/api/v1/health", headers=headers)
                self.assertFalse(private_health.json()["tensorflow_loaded"])
                plan = client.post("/api/v1/batch/plan", headers=headers, json={"dataset": "demo", "profile": "eco", "batch_mode": "manual", "batch_size": 512, "ram": 35})
                self.assertEqual(plan.status_code, 200)
                self.assertTrue(plan.json()["requires_confirmation"])
                blocked = client.post("/api/v1/runs", headers=headers, json={"dataset": "demo", "backend": "numpy", "batch_mode": "manual", "batch_size": 512})
                self.assertEqual(blocked.status_code, 400)


if __name__ == "__main__":
    unittest.main()
