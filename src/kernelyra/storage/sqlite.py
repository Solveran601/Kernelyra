from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any

from ..errors import StorageError
from ..models import DatasetInfo, RunInfo

SCHEMA_VERSION = 4


class SQLiteStorage:
    """SQLite persistence with explicit open and idempotent migrations."""

    def __init__(self, database: Path):
        self.database = database

    @classmethod
    def open(cls, state_dir: Path) -> SQLiteStorage:
        state_dir.mkdir(parents=True, exist_ok=True)
        storage = cls(state_dir / "runs.sqlite3")
        try:
            storage.migrate()
        except sqlite3.DatabaseError as error:
            raise StorageError(
                f"SQLite database is corrupted or unreadable: {type(error).__name__}. Run kernelyra repair."
            ) from None
        return storage

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=15, isolation_level=None)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=15000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def migrate(self) -> None:
        with self.transaction() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS dataset_records(id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS run_records(id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS action_log(id INTEGER PRIMARY KEY AUTOINCREMENT, actor TEXT NOT NULL, action TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS approval_tokens("
                "token_hash TEXT PRIMARY KEY, action TEXT NOT NULL, resource_id TEXT NOT NULL, "
                "expires_at REAL NOT NULL, used_at REAL, created_at REAL NOT NULL)"
            )
            current = db.execute("SELECT COALESCE(MAX(version),0) FROM schema_migrations").fetchone()[0]
            if int(current) > SCHEMA_VERSION:
                raise StorageError(
                    f"Workspace schema {current} is newer than supported schema {SCHEMA_VERSION}"
                )
            if int(current) < 4:
                db.execute("CREATE INDEX IF NOT EXISTS idx_action_log_created ON action_log(created_at,id)")
                db.execute("CREATE INDEX IF NOT EXISTS idx_run_updated ON run_records(updated_at)")
                db.execute("CREATE INDEX IF NOT EXISTS idx_dataset_updated ON dataset_records(updated_at)")
            for version in range(int(current) + 1, SCHEMA_VERSION + 1):
                db.execute(
                    "INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES (?,?)",
                    (version, time.time()),
                )
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            legacy = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='runs'").fetchone()
            empty = db.execute("SELECT 1 FROM run_records LIMIT 1").fetchone() is None
            if legacy and empty:
                for run_id, payload in db.execute("SELECT id,payload FROM runs"):
                    try:
                        normalized = self._normalize_run(json.loads(payload))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    db.execute("INSERT OR IGNORE INTO run_records(id,payload,updated_at) VALUES (?,?,?)", (run_id, json.dumps(normalized, ensure_ascii=False), time.time()))

    @staticmethod
    def _normalize_run(raw: dict[str, Any]) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "backend": "tensorflow", "effective_backend": None, "objective": "binary_classification", "batch_mode": "auto", "batch_size": 32,
            "batch_min": 8, "batch_max": 64, "batch_risk": "safe", "batch_reason": "Исторический запуск",
            "batch_warnings": [], "batch_adjustments": 0, "samples_seen": 0, "eval_count": 0, "best_step": 0,
            "base_run_id": None, "model_path": None, "stop_requested": False, "paused": False,
            "seed": 42,
            "learning_rate": None, "weight_decay": 0.0, "hidden_layers": [],
            "architecture": "auto", "model_format": "auto",
            "precision": "auto", "data_workers": 0, "prefetch": 1,
            "evaluation_interval": None, "min_improvement": 0.0005,
            "degradation_margin": 0.03, "degradation_patience": 3,
            "early_stopping_patience": 18, "target_patience": 3,
        }
        merged = {**defaults, **raw}
        merged.setdefault("target_score", merged.pop("target_metric", .92))
        merged.setdefault("message", "Импортировано из Kernelyra 0.1")
        merged.setdefault("created_at", time.time())
        merged.setdefault("priority", "normal")
        merged.setdefault("profile", "eco")
        merged.setdefault("max_steps", 1000)
        merged.setdefault("cpu", 30)
        merged.setdefault("ram", 35)
        merged.setdefault("gpu", 0)
        merged.setdefault("mode", "Новая модель")
        merged.setdefault("loss", 0.0)
        merged.setdefault("step", 0)
        merged.setdefault("best_score", 0.0)
        allowed = {item.name for item in fields(RunInfo)}
        return {key: value for key, value in merged.items() if key in allowed}

    def save_dataset(self, dataset: DatasetInfo) -> None:
        payload = json.dumps(dataset.to_dict(), ensure_ascii=False)
        with self.transaction() as db:
            db.execute("INSERT INTO dataset_records(id,payload,updated_at) VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at", (dataset.id, payload, time.time()))

    def list_datasets(self) -> list[DatasetInfo]:
        with self.transaction() as db:
            rows = db.execute("SELECT payload FROM dataset_records ORDER BY updated_at DESC").fetchall()
        return [DatasetInfo(**json.loads(row[0])) for row in rows]

    def get_dataset(self, dataset_id: str) -> DatasetInfo | None:
        with self.transaction() as db:
            row = db.execute("SELECT payload FROM dataset_records WHERE id=?", (dataset_id,)).fetchone()
        return DatasetInfo(**json.loads(row[0])) if row else None

    def delete_dataset(self, dataset_id: str) -> None:
        with self.transaction() as db:
            cursor = db.execute("DELETE FROM dataset_records WHERE id=?", (dataset_id,))
            if cursor.rowcount != 1:
                raise KeyError(dataset_id)

    def save_run(self, run: RunInfo) -> None:
        payload = json.dumps(run.to_dict(), ensure_ascii=False)
        with self.transaction() as db:
            db.execute("INSERT INTO run_records(id,payload,updated_at) VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at", (run.id, payload, time.time()))

    def list_runs(self) -> list[RunInfo]:
        with self.transaction() as db:
            rows = db.execute("SELECT payload FROM run_records ORDER BY updated_at DESC").fetchall()
        return [RunInfo(**self._normalize_run(json.loads(row[0]))) for row in rows]

    def get_run(self, run_id: str) -> RunInfo | None:
        with self.transaction() as db:
            row = db.execute("SELECT payload FROM run_records WHERE id=?", (run_id,)).fetchone()
        return RunInfo(**self._normalize_run(json.loads(row[0]))) if row else None

    def delete_run(self, run_id: str) -> None:
        with self.transaction() as db:
            cursor = db.execute("DELETE FROM run_records WHERE id=?", (run_id,))
            if cursor.rowcount != 1:
                raise KeyError(run_id)

    def log_action(self, actor: str, action: str, payload: dict[str, Any]) -> None:
        with self.transaction() as db:
            db.execute("INSERT INTO action_log(actor,action,payload,created_at) VALUES (?,?,?,?)", (actor, action, json.dumps(payload, ensure_ascii=False), time.time()))

    def recent_actions(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.transaction() as db:
            rows = db.execute("SELECT actor,action,payload,created_at FROM action_log ORDER BY id DESC LIMIT ?", (max(1, min(1000, limit)),)).fetchall()
        return [{"actor": actor, "action": action, "payload": json.loads(payload), "created_at": created_at} for actor, action, payload, created_at in rows]

    def events_since(self, last_id: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        with self.transaction() as db:
            rows = db.execute(
                "SELECT id,actor,action,payload,created_at FROM action_log "
                "WHERE id>? ORDER BY id ASC LIMIT ?",
                (max(0, int(last_id)), max(1, min(500, int(limit)))),
            ).fetchall()
        return [
            {
                "id": int(event_id),
                "actor": actor,
                "action": action,
                "payload": json.loads(payload),
                "created_at": created_at,
            }
            for event_id, actor, action, payload, created_at in rows
        ]

    def integrity_check(self) -> dict[str, Any]:
        with self.transaction() as db:
            rows = [str(row[0]) for row in db.execute("PRAGMA integrity_check").fetchall()]
            version = int(db.execute("PRAGMA user_version").fetchone()[0])
            schema = int(
                db.execute("SELECT COALESCE(MAX(version),0) FROM schema_migrations").fetchone()[0]
            )
        return {"ok": rows == ["ok"], "messages": rows[:100], "user_version": version, "schema_version": schema}

    def backup(self, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = self.connect()
        target = sqlite3.connect(destination)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        return destination

    def revoke_approval(self, token: str) -> bool:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = time.time()
        with self.transaction() as db:
            cursor = db.execute(
                "UPDATE approval_tokens SET used_at=? WHERE token_hash=? AND used_at IS NULL",
                (now, token_hash),
            )
        if cursor.rowcount == 1:
            self.log_action("local_user", "approval.revoke", {})
            return True
        return False

    def issue_approval(
        self,
        action: str,
        resource_id: str,
        ttl_seconds: int = 300,
        actor: str = "local_user",
    ) -> tuple[str, float]:
        ttl = max(30, min(3600, int(ttl_seconds)))
        token = "tfap_" + secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = time.time()
        expires_at = now + ttl
        with self.transaction() as db:
            db.execute(
                "INSERT INTO approval_tokens(token_hash,action,resource_id,expires_at,used_at,created_at) "
                "VALUES (?,?,?,?,NULL,?)",
                (token_hash, action, resource_id, expires_at, now),
            )
        self.log_action(
            actor,
            "approval.issue",
            {"action": action, "resource_id": resource_id, "expires_at": expires_at},
        )
        return token, expires_at

    def consume_approval(self, token: str, action: str, resource_id: str) -> bool:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = time.time()
        with self.transaction() as db:
            cursor = db.execute(
                "UPDATE approval_tokens SET used_at=? WHERE token_hash=? AND action=? AND resource_id=? "
                "AND used_at IS NULL AND expires_at>=?",
                (now, token_hash, action, resource_id, now),
            )
            consumed = int(cursor.rowcount) == 1
        if consumed:
            self.log_action("mcp", "approval.consume", {"action": action, "resource_id": resource_id})
        return consumed
