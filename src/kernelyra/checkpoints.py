from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .errors import RunError


class CheckpointManager:
    FORMAT_VERSION = 1

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def best_path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.npz"

    def last_path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.last.npz"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def metadata_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".json")

    def record(self, path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        if path.parent.resolve() != self.root.resolve() or not path.is_file():
            raise RunError("Checkpoint path is outside managed storage")
        payload = {
            **metadata,
            "checkpoint_format": self.FORMAT_VERSION,
            "filename": path.name,
            "sha256": self._sha256(path),
            "size_bytes": path.stat().st_size,
        }
        destination = self.metadata_path(path)
        pending = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.pending")
        with pending.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, destination)
        return payload

    def verify(self, path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
        if path.parent.resolve() != self.root.resolve() or not path.is_file():
            raise RunError("Checkpoint not found in managed storage")
        metadata_path = self.metadata_path(path)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise RunError("Checkpoint metadata is missing or corrupted") from None
        if not isinstance(metadata, dict):
            raise RunError("Checkpoint metadata must be a JSON object")
        if metadata.get("checkpoint_format") != self.FORMAT_VERSION:
            raise RunError("Checkpoint format version is incompatible")
        if metadata.get("sha256") != self._sha256(path):
            raise RunError("Checkpoint checksum mismatch")
        for key, value in (expected or {}).items():
            if value is not None and metadata.get(key) != value:
                raise RunError(f"Checkpoint compatibility mismatch: {key}")
        return {str(key): value for key, value in metadata.items()}

    def promote(self, source: Path, destination: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        if source.parent.resolve() != self.root.resolve() or destination.parent.resolve() != self.root.resolve():
            raise RunError("Checkpoint promotion escaped managed storage")
        pending = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.pending")
        with source.open("rb") as input_file, pending.open("xb") as output_file:
            shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(pending, destination)
        return self.record(destination, metadata)

    def remove_run(self, run_id: str) -> None:
        for path in (self.best_path(run_id), self.last_path(run_id)):
            path.unlink(missing_ok=True)
            self.metadata_path(path).unlink(missing_ok=True)

    def discard_last(self, run_id: str) -> None:
        path = self.last_path(run_id)
        path.unlink(missing_ok=True)
        self.metadata_path(path).unlink(missing_ok=True)
