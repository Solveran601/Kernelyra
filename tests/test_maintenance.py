from __future__ import annotations

import json
import unittest

from kernelyra.maintenance import (
    cleanup_workspace,
    export_workspace_manifest,
    import_workspace_manifest,
    inspect_workspace,
    migrate_workspace,
    repair_workspace,
    validate_config,
)
from kernelyra.storage import SQLiteStorage
from tests.helpers import isolated_workspace


class MaintenanceTests(unittest.TestCase):
    def test_migrate_backup_integrity_and_config_validation(self) -> None:
        with isolated_workspace() as temporary:
            root = temporary / "проект с пробелом"
            first = migrate_workspace(root)
            self.assertTrue(first["integrity"]["ok"])
            second = migrate_workspace(root)
            self.assertTrue(second["integrity"]["ok"])
            self.assertIsNotNone(second["backup"])
            config = temporary / "agent.json"
            config.write_text(json.dumps({"allowed_roots": [str(root)]}), encoding="utf-8")
            self.assertTrue(validate_config(config)["ok"])

    def test_repair_is_dry_run_and_quarantines_orphans_only_with_apply(self) -> None:
        with isolated_workspace() as temporary:
            root = temporary / "project"
            storage = SQLiteStorage.open(root / ".kernelyra")
            orphan = root / ".kernelyra" / "checkpoints" / "orphan.npz"
            orphan.parent.mkdir(parents=True)
            orphan.write_bytes(b"checkpoint")
            pid = root / ".kernelyra" / "daemon.pid"
            pid.write_text("99999999", encoding="ascii")
            report = inspect_workspace(root)
            self.assertTrue(orphan.exists())
            self.assertTrue(any(item["kind"] == "orphan_checkpoint" for item in report["findings"]))
            dry_run = repair_workspace(root)
            self.assertFalse(dry_run["changes_applied"])
            applied = repair_workspace(root, apply=True)
            self.assertTrue(applied["changes_applied"])
            self.assertFalse(orphan.exists())
            self.assertFalse(pid.exists())
            self.assertTrue(storage.integrity_check()["ok"])

    def test_cleanup_only_removes_pending_files_on_apply(self) -> None:
        with isolated_workspace() as temporary:
            pending = temporary / "project" / ".kernelyra" / "uploads" / "pending-file.csv"
            pending.parent.mkdir(parents=True)
            pending.write_text("x", encoding="utf-8")
            self.assertEqual(len(cleanup_workspace(temporary / "project")["candidates"]), 1)
            self.assertTrue(pending.exists())
            cleanup_workspace(temporary / "project", apply=True)
            self.assertFalse(pending.exists())

    def test_workspace_manifest_restores_metadata_without_secrets_or_locks(self) -> None:
        with isolated_workspace() as temporary:
            root = temporary / "project"
            from kernelyra import Workspace

            workspace = Workspace.open(root)
            workspace.datasets.get("demo")
            manifest = export_workspace_manifest(root, temporary / "workspace.json")
            workspace.close()
            text = manifest.read_text(encoding="utf-8")
            self.assertNotIn("daemon.secret", text)
            self.assertNotIn("daemon.lock", text)
            for suffix in ("", "-wal", "-shm"):
                (root / ".kernelyra" / f"runs.sqlite3{suffix}").unlink(missing_ok=True)
            restored = import_workspace_manifest(root, manifest)
            self.assertGreaterEqual(restored["datasets"], 1)
            storage = SQLiteStorage.open(root / ".kernelyra")
            self.assertIsNotNone(storage.get_dataset("demo"))


if __name__ == "__main__":
    unittest.main()
