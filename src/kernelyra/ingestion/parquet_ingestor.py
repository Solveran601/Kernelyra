from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import DatasetError
from .tabular import prepare_rows


def _parquet_file(path: Path) -> Any:
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError:
        raise DatasetError(
            'Parquet распознан, но optional dependency отсутствует: pip install "kernelyra-ai[parquet]"'
        ) from None
    try:
        return parquet.ParquetFile(path)
    except Exception as error:
        raise DatasetError(f"Повреждённый или несовместимый Parquet: {error}") from None


class ParquetIngestor:
    name = "parquet"
    version = "1.0"
    extensions = (".parquet", ".pq")
    task_types = ("binary_classification", "multiclass_classification", "regression")

    @staticmethod
    def available() -> bool:
        try:
            import pyarrow  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    @staticmethod
    def inspect(path: Path) -> dict[str, Any]:
        parquet = _parquet_file(path)
        columns = list(parquet.schema.names)
        preview: list[dict[str, Any]] = []
        for batch in parquet.iter_batches(batch_size=20):
            preview.extend(batch.to_pylist())
            break
        return {
            "columns": columns,
            "suggested_target": columns[-1] if columns else None,
            "rows": int(parquet.metadata.num_rows),
            "row_groups": int(parquet.metadata.num_row_groups),
            "preview": preview[:20],
            "streaming": True,
            "optional_dependency": "pyarrow",
        }

    @staticmethod
    def import_file(
        path: Path, target: str | None = None
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        parquet = _parquet_file(path)
        columns = list(parquet.schema.names)

        def rows() -> Iterator[dict[str, Any]]:
            for batch in parquet.iter_batches(batch_size=4096):
                yield from batch.to_pylist()

        prepared = prepare_rows(columns, rows(), target)
        return {
            **prepared.metadata,
            "validation_report": {
                "valid": True,
                "warnings": prepared.metadata["warnings"],
                "skipped_rows": prepared.metadata["skipped"],
            },
            "streaming": True,
            "optional_dependency": "pyarrow",
        }, prepared.x, prepared.y
