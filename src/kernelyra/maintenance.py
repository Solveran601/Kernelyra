from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, StorageError
from .models import DatasetInfo, RunInfo
from .storage import SQLiteStorage


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def validate_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"Invalid JSON configuration: {error}") from None
    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration root must be a JSON object")
    allowed = {"allowed_roots", "allowed_actions", "daemon_url", "client_id", "timeout"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigurationError(f"Unknown configuration fields: {', '.join(unknown)}")
    if "allowed_roots" in raw and not isinstance(raw["allowed_roots"], list):
        raise ConfigurationError("allowed_roots must be a list")
    if "allowed_actions" in raw and not isinstance(raw["allowed_actions"], list):
        raise ConfigurationError("allowed_actions must be a list")
    return {"ok": True, "path": str(config_path), "fields": sorted(raw)}


def migrate_workspace(root: str | Path) -> dict[str, Any]:
    workspace = Path(root).expanduser().resolve()
    state = workspace / ".kernelyra"
    state.mkdir(parents=True, exist_ok=True)
    database = state / "runs.sqlite3"
    backup: Path | None = None
    storage = SQLiteStorage(database)
    if database.exists() and database.stat().st_size:
        backup = state / "backups" / f"runs-before-migrate-{int(time.time())}.sqlite3"
        try:
            storage.backup(backup)
        except Exception as error:
            raise StorageError(f"Could not create migration backup: {error}") from None
    storage.migrate()
    return {
        "ok": True,
        "workspace": str(workspace),
        "backup": str(backup) if backup else None,
        "integrity": storage.integrity_check(),
    }


def _record_ids(storage: SQLiteStorage) -> tuple[set[str], set[str]]:
    return ({item.id for item in storage.list_datasets()}, {item.id for item in storage.list_runs()})


def inspect_workspace(root: str | Path) -> dict[str, Any]:
    workspace = Path(root).expanduser().resolve()
    state = workspace / ".kernelyra"
    findings: list[dict[str, Any]] = []
    pid_path = state / "daemon.pid"
    pid: int | None = None
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            findings.append({"kind": "invalid_pid", "path": str(pid_path), "safe_to_repair": True})
        else:
            if process_alive(pid):
                findings.append({"kind": "live_daemon", "pid": pid, "safe_to_repair": False})
            else:
                findings.append({"kind": "stale_pid", "pid": pid, "path": str(pid_path), "safe_to_repair": True})

    database = state / "runs.sqlite3"
    integrity: dict[str, Any] = {"ok": True, "status": "not_created"}
    dataset_ids: set[str] = set()
    run_ids: set[str] = set()
    if database.exists():
        storage = SQLiteStorage(database)
        try:
            integrity = storage.integrity_check()
            if integrity["ok"]:
                dataset_ids, run_ids = _record_ids(storage)
        except Exception as error:
            integrity = {"ok": False, "error": type(error).__name__, "message": str(error)[:300]}
            findings.append({"kind": "sqlite_corruption", "path": str(database), "safe_to_repair": False})

    for folder, suffix, known, kind in (
        (state / "datasets", "", dataset_ids, "orphan_dataset"),
        (state / "arrays", ".npz", dataset_ids, "orphan_array"),
        (state / "checkpoints", ".npz", run_ids, "orphan_checkpoint"),
    ):
        if not folder.is_dir():
            continue
        for candidate in folder.iterdir():
            if not candidate.is_file():
                continue
            identifier = candidate.stem
            if kind == "orphan_checkpoint":
                identifier = identifier.removesuffix(".last")
            if suffix and candidate.suffix.lower() != suffix:
                continue
            belongs_to_record = identifier in known
            if kind == "orphan_dataset":
                belongs_to_record = any(candidate.name.startswith(f"{item}_") for item in known)
            if not belongs_to_record:
                findings.append({"kind": kind, "path": str(candidate), "safe_to_repair": True})
    return {
        "workspace": str(workspace),
        "daemon_pid": pid,
        "integrity": integrity,
        "findings": findings,
        "changes_applied": False,
    }


def repair_workspace(root: str | Path, *, apply: bool = False) -> dict[str, Any]:
    report = inspect_workspace(root)
    if not apply:
        return report
    if any(item["kind"] == "live_daemon" for item in report["findings"]):
        raise ConfigurationError("Repair refused while the workspace daemon is running")
    state = Path(report["workspace"]) / ".kernelyra"
    recovery = state / "recovery" / time.strftime("%Y%m%d-%H%M%S")
    changed: list[dict[str, str]] = []
    for item in report["findings"]:
        if not item.get("safe_to_repair"):
            continue
        path_value = item.get("path")
        if not path_value:
            continue
        source = Path(path_value)
        if not source.exists() or state not in source.parents:
            continue
        if item["kind"] in {"stale_pid", "invalid_pid"}:
            source.unlink(missing_ok=True)
            changed.append({"kind": item["kind"], "action": "removed", "path": str(source)})
            for metadata_name in ("daemon.json", "daemon.lock"):
                metadata = state / metadata_name
                metadata.unlink(missing_ok=True)
            continue
        recovery.mkdir(parents=True, exist_ok=True)
        destination = recovery / source.name
        index = 1
        while destination.exists():
            destination = recovery / f"{source.stem}-{index}{source.suffix}"
            index += 1
        source.replace(destination)
        changed.append({"kind": item["kind"], "action": "quarantined", "path": str(source), "destination": str(destination)})
    report["changes_applied"] = True
    report["changes"] = changed
    report["recovery_directory"] = str(recovery) if changed else None
    return report


def cleanup_workspace(root: str | Path, *, apply: bool = False) -> dict[str, Any]:
    workspace = Path(root).expanduser().resolve()
    state = workspace / ".kernelyra"
    candidates: list[Path] = []
    for folder in (state / "uploads", state / "datasets", state / "arrays", state / "checkpoints"):
        if folder.is_dir():
            candidates.extend(
                item for item in folder.iterdir()
                if item.is_file() and (item.name.startswith("pending-") or ".pending" in item.name)
            )
    report = {"workspace": str(workspace), "candidates": [str(item) for item in candidates], "applied": apply}
    if apply:
        for item in candidates:
            item.unlink(missing_ok=True)
    return report


def export_workspace_manifest(root: str | Path, destination: str | Path) -> Path:
    workspace = Path(root).expanduser().resolve()
    state = workspace / ".kernelyra"
    storage = SQLiteStorage(state / "runs.sqlite3")
    payload = {
        "contract_version": "kernelyra-workspace-manifest/1",
        "created_at": time.time(),
        "datasets": [
            {
                **{key: value for key, value in item.to_dict().items() if key != "path"},
                "storage_relative": Path(item.path).resolve().relative_to(state.resolve()).as_posix(),
                "array_filename": f"{item.id}.npz",
            }
            for item in storage.list_datasets()
        ],
        "runs": [
            {
                **{key: value for key, value in item.to_dict().items() if key != "model_path"},
                "model_filename": Path(item.model_path).name if item.model_path else None,
            }
            for item in storage.list_runs()
        ],
    }
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(f".{output.name}.pending")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(pending, output)
    return output


def import_workspace_manifest(root: str | Path, source: str | Path) -> dict[str, Any]:
    workspace = Path(root).expanduser().resolve()
    state = workspace / ".kernelyra"
    manifest_path = Path(source).expanduser().resolve()
    if manifest_path.stat().st_size > 10 * 1024 * 1024:
        raise ConfigurationError("Workspace manifest exceeds 10 MB")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"Invalid workspace manifest: {error}") from None
    if payload.get("contract_version") != "kernelyra-workspace-manifest/1":
        raise ConfigurationError("Workspace manifest contract is incompatible")
    storage = SQLiteStorage.open(state)
    imported_datasets = 0
    imported_runs = 0
    for raw in payload.get("datasets", []):
        if not isinstance(raw, dict):
            raise ConfigurationError("Invalid dataset record in workspace manifest")
        relative_name = str(raw.pop("storage_relative", ""))
        array_filename = str(raw.pop("array_filename", ""))
        relative = Path(relative_name.replace("/", os.sep))
        if relative.is_absolute() or ".." in relative.parts or Path(array_filename).name != array_filename:
            raise ConfigurationError("Workspace manifest contains an unsafe storage filename")
        dataset_path = (state / relative).resolve()
        if state.resolve() not in dataset_path.parents:
            raise ConfigurationError("Workspace manifest dataset path escaped managed storage")
        array_path = state / "arrays" / array_filename
        if not dataset_path.is_file() or not array_path.is_file():
            raise ConfigurationError(f"Managed files for dataset {raw.get('id')} are missing")
        if _sha256(dataset_path) != raw.get("sha256"):
            raise ConfigurationError(f"Dataset checksum mismatch for {raw.get('id')}")
        dataset = DatasetInfo(**raw, path=str(dataset_path))
        storage.save_dataset(dataset)
        imported_datasets += 1
    for raw in payload.get("runs", []):
        if not isinstance(raw, dict):
            raise ConfigurationError("Invalid run record in workspace manifest")
        raw.pop("model_filename", None)
        raw["model_path"] = None
        if raw.get("status") in {"queued", "training", "pausing", "stopping"}:
            raw["status"] = "error_recoverable"
            raw["termination_reason"] = "manifest_restore"
        run = RunInfo(**SQLiteStorage._normalize_run(raw))
        storage.save_run(run)
        imported_runs += 1
    storage.log_action(
        "maintenance",
        "workspace.manifest.import",
        {"datasets": imported_datasets, "runs": imported_runs},
    )
    return {
        "ok": True,
        "workspace": str(workspace),
        "datasets": imported_datasets,
        "runs": imported_runs,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
