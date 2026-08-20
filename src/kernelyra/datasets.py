from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .errors import DatasetError, DatasetNotFoundError
from .ingestion.registry import IngestorRegistry
from .ingestion.router import FormatRouter
from .models import DatasetInfo, DatasetManifest
from .storage import SQLiteStorage
from .streaming import build_stream_spec


class DatasetManager:
    MAX_IMPORT_BYTES = 512 * 1024 * 1024

    def __init__(
        self,
        root: Path,
        state_dir: Path,
        storage: SQLiteStorage,
        native_probe: Path | None = None,
    ):
        self.root = root
        self.state_dir = state_dir
        self.files_dir = state_dir / "datasets"
        self.arrays_dir = state_dir / "arrays"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        self.storage = storage
        self.ingestors = IngestorRegistry()
        self.router = FormatRouter(native_probe, self.ingestors)
        self._migrate_legacy_index()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _ensure_demo(self) -> None:
        if self.storage.get_dataset("demo"):
            return
        path = self.state_dir / "demo_dataset.csv"
        if not path.exists():
            rng = np.random.default_rng(42)
            x = rng.normal(size=(2400, 4))
            y = (
                (x[:, 0] * 1.5 - x[:, 1] + .55 * x[:, 2] + rng.normal(0, .7, len(x))) > 0
            ).astype(int)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["feature_1", "feature_2", "feature_3", "feature_4", "target"])
                writer.writerows([*row, int(label)] for row, label in zip(x, y, strict=False))
        self._register(
            path,
            "target",
            "demo",
            "Built-in synthetic dataset",
            source_kind="generated",
            content_hash=self._sha256(path),
        )

    def _migrate_legacy_index(self) -> None:
        index = self.state_dir / "datasets.json"
        if not index.exists():
            return
        try:
            items = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for item in items:
            if self.storage.get_dataset(str(item.get("id", ""))):
                continue
            path = self.files_dir / str(item.get("file", ""))
            if not path.exists():
                continue
            try:
                self._register(
                    path,
                    item.get("target"),
                    str(item["id"]),
                    str(item.get("source") or path.name),
                    source_kind="legacy",
                    content_hash=self._sha256(path),
                )
            except (DatasetError, KeyError, OSError):
                continue

    def inspect(self, path: str | Path) -> dict[str, Any]:
        return self.router.inspect(path)

    def import_file(
        self,
        path: str | Path,
        target: str | None = None,
        source_name: str | None = None,
        *,
        cancel_check: Callable[[], bool] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> DatasetInfo:
        requested = Path(path).expanduser()
        if requested.is_symlink():
            raise DatasetError("Импорт через symbolic link/reparse point запрещён")
        source = requested.resolve()
        if not source.exists() or not source.is_file():
            raise DatasetError("Файл датасета не найден")
        size = source.stat().st_size
        if size > self.MAX_IMPORT_BYTES:
            raise DatasetError("Файл больше безопасного лимита импорта 512 MB")
        ingestor = self.ingestors.for_path(source)
        if ingestor is None:
            raise DatasetError("Формат распознан, но обучающий ingestor для него не установлен")

        inspected = ingestor.inspect(source)
        selected_target = target or inspected.get("suggested_target")
        content_hash = self._sha256(source)
        for existing in self.storage.list_datasets():
            if existing.sha256 == content_hash and existing.target == selected_target:
                return existing

        original_name = Path(source_name or source.name).name
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", original_name) or "dataset"
        target_key = hashlib.sha256(str(selected_target).encode("utf-8")).hexdigest()[:4]
        dataset_id = f"ds_{content_hash[:12]}_{target_key}"
        copied = self.files_dir / f"{dataset_id}_{safe_name}"
        pending = self.files_dir / f".pending-{uuid.uuid4().hex}-{safe_name}"
        try:
            copied_bytes = 0
            with source.open("rb") as input_file, pending.open("xb") as output_file:
                while True:
                    if cancel_check and cancel_check():
                        raise DatasetError("Импорт отменён пользователем")
                    chunk = input_file.read(1024 * 1024)
                    if not chunk:
                        break
                    output_file.write(chunk)
                    copied_bytes += len(chunk)
                    if progress:
                        progress(copied_bytes, size)
                output_file.flush()
                os.fsync(output_file.fileno())
            if self._sha256(pending) != content_hash:
                raise DatasetError("Контрольная сумма изменилась во время копирования")
            os.replace(pending, copied)
            return self._register(
                copied,
                selected_target,
                dataset_id,
                original_name,
                source_kind="file",
                content_hash=content_hash,
            )
        except Exception:
            pending.unlink(missing_ok=True)
            copied.unlink(missing_ok=True)
            raise

    def attach_path(self, path: str | Path, target: str | None = None) -> DatasetInfo:
        """Register a file or folder by reference without copying or materializing it."""
        source = Path(path).expanduser().resolve()
        spec = build_stream_spec(source, target)
        fingerprint = str(spec["fingerprint"])
        target_key = hashlib.sha256(str(spec["target"]).encode("utf-8")).hexdigest()[:4]
        dataset_id = f"stream_{fingerprint[:12]}_{target_key}"
        existing = self.storage.get_dataset(dataset_id)
        if existing:
            return existing
        created_at = time.time()
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            source_name=source.name,
            source_kind="external_stream_folder" if source.is_dir() else "external_stream",
            format=str(spec["format"]),
            sha256=fingerprint,
            size_bytes=int(spec["size_bytes"]),
            row_count=int(spec["records"]),
            schema={
                "columns": spec["columns"],
                "target": spec["target"],
                "task_types": [spec["task_type"]],
                "feature_count": spec["features"],
                "row_count": spec["records"],
            },
            task_compatibility=(str(spec["task_type"]),),
            transformations=(
                {"kind": "standardize", "columns": spec["numeric_columns"]},
                {"kind": "feature_hash", "columns": spec["categorical_columns"], "buckets": spec["hash_buckets"]},
            ),
            split_seed=42,
            warnings=("External source files must remain available and unchanged",),
            created_at=created_at,
            ingestor_name=f"stream-{spec['reader_engine']}",
            ingestor_version="1",
        )
        info = DatasetInfo(
            id=dataset_id,
            source=source.name,
            path=str(source),
            records=int(spec["records"]),
            features=int(spec["features"]),
            target=str(spec["target"]),
            classes=list(spec["classes"]),
            format=str(spec["format"]),
            sha256=fingerprint,
            size_bytes=int(spec["size_bytes"]),
            task_types=[str(spec["task_type"])],
            schema=dict(manifest.schema),
            manifest={**manifest.to_dict(), "streaming": spec},
            warnings=list(manifest.warnings),
            created_at=created_at,
        )
        self.storage.save_dataset(info)
        return info

    def attach_file(self, path: str | Path, target: str | None = None) -> DatasetInfo:
        """Backward-compatible alias for :meth:`attach_path`."""
        return self.attach_path(path, target)

    def _register(
        self,
        path: Path,
        target: str | None,
        dataset_id: str,
        source: str,
        *,
        source_kind: str,
        content_hash: str,
    ) -> DatasetInfo:
        ingestor = self.ingestors.for_path(path)
        if ingestor is None:
            raise DatasetError("Обучающий ingestor для формата не установлен")
        metadata, x, y = ingestor.import_file(path, target)
        metadata.setdefault("format", ingestor.name)
        array_path = self.arrays_dir / f"{dataset_id}.npz"
        pending_array = self.arrays_dir / f".{dataset_id}.{uuid.uuid4().hex}.pending.npz"
        np.savez_compressed(pending_array, x=x, y=y)
        os.replace(pending_array, array_path)
        created_at = time.time()
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            source_name=Path(source).name,
            source_kind=source_kind,
            format=str(metadata["format"]),
            sha256=content_hash,
            size_bytes=path.stat().st_size,
            row_count=int(metadata["records"]),
            schema=dict(metadata.get("schema", {})),
            task_compatibility=tuple(metadata.get("task_types", ())),
            transformations=tuple(metadata.get("transformations", ())),
            split_seed=42,
            warnings=tuple(metadata.get("warnings", ())),
            created_at=created_at,
            ingestor_name=str(ingestor.name),
            ingestor_version=str(getattr(ingestor, "version", "0")),
        )
        info = DatasetInfo(
            id=dataset_id,
            source=source,
            path=str(path),
            records=int(metadata["records"]),
            features=int(metadata["features"]),
            target=str(metadata["target"]),
            classes=list(metadata.get("classes", [])),
            format=str(metadata["format"]),
            skipped=int(metadata.get("skipped", 0)),
            sha256=content_hash,
            size_bytes=path.stat().st_size,
            task_types=list(metadata.get("task_types", [])),
            schema=dict(metadata.get("schema", {})),
            manifest=manifest.to_dict(),
            warnings=list(metadata.get("warnings", [])),
            created_at=created_at,
        )
        self.storage.save_dataset(info)
        return info

    def list(self) -> list[DatasetInfo]:
        return self.storage.list_datasets()

    def get(self, dataset_id: str) -> DatasetInfo:
        dataset = self.storage.get_dataset(dataset_id)
        if not dataset and dataset_id == "demo":
            # Keep real-data workspaces free of synthetic artifacts. The
            # example dataset remains available, but is generated on demand.
            self._ensure_demo()
            dataset = self.storage.get_dataset(dataset_id)
        if not dataset:
            raise DatasetNotFoundError("Датасет не найден")
        return dataset

    def remove(self, dataset_id: str) -> None:
        dataset = self.get(dataset_id)
        if dataset.id == "demo":
            raise DatasetError("Встроенный demo dataset удалить нельзя")
        path = Path(dataset.path)
        external = str(dataset.manifest.get("source_kind", "")).startswith("external_stream")
        if not external and path.parent.resolve() != self.files_dir.resolve():
            raise DatasetError("Dataset path вышел за пределы runtime storage")
        self.storage.delete_dataset(dataset_id)
        if not external:
            path.unlink(missing_ok=True)
        (self.arrays_dir / f"{dataset_id}.npz").unlink(missing_ok=True)

    def load_arrays(self, dataset_id: str) -> tuple[np.ndarray, np.ndarray]:
        dataset = self.get(dataset_id)
        if str(dataset.manifest.get("source_kind", "")).startswith("external_stream"):
            raise DatasetError("External streaming datasets do not materialize complete runtime arrays")
        array_path = self.arrays_dir / f"{dataset_id}.npz"
        if not array_path.exists():
            ingestor = self.ingestors.for_path(Path(dataset.path))
            if ingestor is None:
                raise DatasetError(f"Ingestor для датасета '{dataset.format}' не установлен")
            _, x, y = ingestor.import_file(Path(dataset.path), dataset.target)
            pending = self.arrays_dir / f".{dataset_id}.{uuid.uuid4().hex}.pending.npz"
            np.savez_compressed(pending, x=x, y=y)
            os.replace(pending, array_path)
            return x, y
        try:
            with np.load(array_path, allow_pickle=False) as saved:
                return saved["x"].copy(), saved["y"].copy()
        except (OSError, ValueError, KeyError) as error:
            raise DatasetError(f"Runtime arrays повреждены: {error}") from None
