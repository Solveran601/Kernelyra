from __future__ import annotations

import sqlite3
import threading
import time
import unittest
import zipfile
from unittest.mock import patch

from kernelyra.checkpoints import CheckpointManager
from kernelyra.errors import ConfigurationError, DatasetError, RunError
from kernelyra.ingestion.npz_ingestor import NPZIngestor
from kernelyra.maintenance import repair_workspace
from kernelyra.models import RunConfig
from kernelyra.workspace import Workspace
from tests.helpers import isolated_workspace


class FailureModeTests(unittest.TestCase):
    def test_corrupted_checkpoint_is_rejected_without_deleting_it(self) -> None:
        with isolated_workspace() as temporary:
            manager = CheckpointManager(temporary / "checkpoints")
            checkpoint = manager.best_path("run1")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"not-a-checkpoint")
            with self.assertRaises(RunError):
                manager.verify(checkpoint, {})
            self.assertTrue(checkpoint.exists())

    def test_stale_scheduler_lock_is_repaired_but_live_runtime_is_not_taken_over(self) -> None:
        with isolated_workspace() as temporary:
            root = temporary / "project"
            stale = root / ".kernelyra" / "daemon.pid"
            stale.parent.mkdir(parents=True)
            stale.write_text("99999999", encoding="ascii")
            repaired = repair_workspace(root, apply=True)
            self.assertTrue(repaired["changes_applied"])
            first = Workspace.open(root)
            second = Workspace.open(root)
            first.runtime.start()
            with self.assertRaises(ConfigurationError):
                second.runtime.start()
            first.close()
            second.close()

    def test_sqlite_busy_writer_recovers_after_short_contention(self) -> None:
        with isolated_workspace() as temporary:
            workspace = Workspace.open(temporary / "project")
            database = workspace.state_dir / "runs.sqlite3"
            blocker = sqlite3.connect(database, isolation_level=None)
            blocker.execute("BEGIN IMMEDIATE")
            outcome: list[str] = []

            def writer() -> None:
                workspace.storage.log_action("test", "sqlite.concurrent", {})
                outcome.append("written")

            thread = threading.Thread(target=writer)
            thread.start()
            time.sleep(.2)
            blocker.execute("ROLLBACK")
            blocker.close()
            thread.join(timeout=5)
            self.assertEqual(outcome, ["written"])
            workspace.close()

    def test_unicode_space_workspace_and_dataset_path(self) -> None:
        with isolated_workspace() as temporary:
            root = temporary / "Проект с пробелами"
            source = temporary / "данные для обучения.csv"
            source.write_text("x,target\n" + "\n".join(f"{i},{i % 2}" for i in range(80)), encoding="utf-8")
            workspace = Workspace.open(root)
            dataset = workspace.datasets.import_file(source, "target")
            run = workspace.create_run(RunConfig(dataset=dataset.id, backend="numpy")).info
            self.assertEqual(run.dataset, dataset.id)
            workspace.close()

    def test_npz_decompression_bomb_is_rejected_before_numpy_load(self) -> None:
        with isolated_workspace() as temporary:
            archive = temporary / "bomb.npz"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
                output.writestr("x.npy", b"0" * (4 * 1024 * 1024))
                output.writestr("y.npy", b"0" * (4 * 1024 * 1024))
            with patch("kernelyra.ingestion.npz_ingestor.np.load") as loader:
                with self.assertRaisesRegex(DatasetError, "decompression bomb"):
                    NPZIngestor.inspect(archive)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
