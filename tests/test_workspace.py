from __future__ import annotations

import json
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path

from kernelyra import RunConfig, Workspace
from kernelyra.errors import ConfigurationError
from tests.helpers import isolated_workspace

ROOT = Path(__file__).resolve().parents[1]


class WorkspaceTests(unittest.TestCase):
    def test_explicit_open_creates_workspace_and_demo_lazily(self) -> None:
        with isolated_workspace() as temporary:
            root = temporary / "project"
            self.assertFalse(root.exists())
            workspace = Workspace.open(root)
            self.assertTrue((root / ".kernelyra" / "runs.sqlite3").exists())
            self.assertFalse((root / ".kernelyra" / "demo_dataset.csv").exists())
            self.assertEqual(workspace.datasets.get("demo").records, 2400)

    def test_create_is_draft_and_does_not_start_thread(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            run = workspace.create_run(RunConfig(dataset="demo", backend="numpy", name="draft-check"))
            self.assertEqual(run.info.status, "draft")
            self.assertIsNone(workspace._runtime)

    def test_unsafe_batch_is_blocked(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            with self.assertRaises(ConfigurationError):
                workspace.create_run(RunConfig(dataset="demo", backend="numpy", batch_mode="manual", batch_size=512))

    def test_legacy_runs_are_migrated(self) -> None:
        with isolated_workspace() as temporary:
            root = temporary / "project"
            state = root / ".kernelyra"
            state.mkdir(parents=True)
            db_path = state / "runs.sqlite3"
            with closing(sqlite3.connect(db_path)) as db:
                with db:
                    db.execute("CREATE TABLE runs(id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
                    payload = {"id": "legacy001", "name": "legacy", "dataset": "demo", "mode": "Новая модель", "gpu": 0, "ram": 35, "cpu": 30, "status": "completed", "step": 100, "max_steps": 100, "best_score": .8, "target_score": .9, "loss": .2}
                    db.execute("INSERT INTO runs VALUES (?,?)", ("legacy001", json.dumps(payload)))
            workspace = Workspace.open(root)
            migrated = workspace.storage.get_run("legacy001")
            self.assertIsNotNone(migrated)
            self.assertEqual(migrated.backend, "tensorflow")


if __name__ == "__main__":
    unittest.main()
