from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import DatasetError
from .csv_ingestor import CSVIngestor
from .jsonl_ingestor import JSONLIngestor
from .npz_ingestor import NPZIngestor
from .parquet_ingestor import ParquetIngestor


class IngestorRegistry:
    """Closed registry of reviewed ingestors shipped with Kernelyra."""

    def __init__(self) -> None:
        self._ingestors: dict[str, Any] = {}
        self._register("csv", CSVIngestor)
        self._register("jsonl", JSONLIngestor)
        self._register("npz", NPZIngestor)
        self._register("parquet", ParquetIngestor)

    def _register(self, name: str, factory: Any) -> None:
        if name in self._ingestors:
            return
        instance = factory() if isinstance(factory, type) else factory
        extensions = tuple(str(item).lower() for item in getattr(instance, "extensions", ()))
        if not extensions or not callable(getattr(instance, "inspect", None)) or not callable(getattr(instance, "import_file", None)):
            raise DatasetError(f"Built-in ingestor '{name}' does not implement the required contract")
        self._ingestors[name] = instance

    def names(self) -> list[str]:
        return sorted(self._ingestors)

    def get(self, name: str) -> Any:
        try:
            return self._ingestors[name]
        except KeyError:
            raise DatasetError(f"Ingestor '{name}' не установлен") from None

    def for_path(self, path: Path) -> Any | None:
        suffix = path.suffix.lower()
        for ingestor in self._ingestors.values():
            if suffix in tuple(str(item).lower() for item in ingestor.extensions):
                return ingestor
        return None

    def describe(self) -> list[dict[str, Any]]:
        descriptions: list[dict[str, Any]] = []
        for name in self.names():
            ingestor = self._ingestors[name]
            checker = getattr(ingestor, "available", None)
            available = bool(checker()) if callable(checker) else True
            descriptions.append(
                {
                    "name": name,
                    "version": str(getattr(ingestor, "version", "0")),
                    "extensions": list(ingestor.extensions),
                    "task_types": list(getattr(ingestor, "task_types", ())),
                    "available": available,
                    "diagnostic": None
                    if available
                    else 'Install optional dependency: pip install "kernelyra-ai[parquet]"',
                }
            )
        return descriptions
