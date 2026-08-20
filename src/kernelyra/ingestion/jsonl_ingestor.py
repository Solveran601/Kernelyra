from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import DatasetError
from .tabular import prepare_rows


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if len(line) > 2 * 1024 * 1024:
                raise DatasetError(f"JSONL line {line_number}: превышен лимит строки 2 MB")
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetError(f"JSONL line {line_number}: {error.msg}") from None
            if not isinstance(value, dict) or any(isinstance(item, dict | list) for item in value.values()):
                raise DatasetError(
                    f"JSONL line {line_number}: core ingestor принимает только плоские объекты со scalar values"
                )
            yield value


class JSONLIngestor:
    name = "jsonl"
    version = "1.0"
    extensions = (".jsonl", ".ndjson")
    task_types = ("binary_classification", "multiclass_classification", "regression")

    @staticmethod
    def inspect(path: Path) -> dict[str, Any]:
        preview: list[dict[str, Any]] = []
        columns: list[str] = []
        for row in _rows(path):
            for key in row:
                if key not in columns:
                    columns.append(key)
            if len(preview) < 20:
                preview.append(row)
            if len(preview) >= 20 and len(columns) >= 2:
                break
        if len(columns) < 2:
            raise DatasetError("JSONL должен содержать хотя бы один признак и target")
        preferred = {"target", "label", "class", "y", "answer", "result"}
        target = next((name for name in columns if name.lower() in preferred), columns[-1])
        return {
            "columns": columns,
            "target_candidates": columns,
            "suggested_target": target,
            "preview": preview,
            "streaming": True,
        }

    @staticmethod
    def import_file(
        path: Path, target: str | None = None
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        iterator = _rows(path)
        try:
            first = next(iterator)
        except StopIteration:
            raise DatasetError("JSONL пуст") from None
        columns = list(first)

        def combined() -> Iterator[dict[str, Any]]:
            yield first
            for row in iterator:
                unknown = set(row) - set(columns)
                if unknown:
                    raise DatasetError(
                        "JSONL schema меняется между строками; новые поля: " + ", ".join(sorted(unknown))
                    )
                yield row

        prepared = prepare_rows(columns, combined(), target)
        return {
            **prepared.metadata,
            "validation_report": {
                "valid": True,
                "warnings": prepared.metadata["warnings"],
                "skipped_rows": prepared.metadata["skipped"],
            },
            "streaming": True,
        }, prepared.x, prepared.y
