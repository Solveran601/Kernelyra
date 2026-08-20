from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from ..errors import DatasetError
from ..models import TaskType
from ..native_core import NativeCoreError, NativeNumericCsv
from .tabular import _number, prepare_rows

MAX_SNIFF_BYTES = 64 * 1024
MAX_FIELD_BYTES = 1024 * 1024
SUPPORTED_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251")
SUPPORTED_DELIMITERS = ",;\t|"


def _open_detected(path: Path) -> tuple[TextIO, str, str]:
    csv.field_size_limit(MAX_FIELD_BYTES)
    with path.open("rb") as binary:
        sample_bytes = binary.read(MAX_SNIFF_BYTES)
    text = ""
    encoding = ""
    for candidate in SUPPORTED_ENCODINGS:
        try:
            text = sample_bytes.decode(candidate)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    if not encoding:
        raise DatasetError("CSV encoding не поддерживается; используйте UTF-8 или Windows-1251")
    try:
        dialect = csv.Sniffer().sniff(text, delimiters=SUPPORTED_DELIMITERS)
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    return path.open("r", newline="", encoding=encoding), encoding, delimiter


class CSVIngestor:
    name = "csv"
    version = "1.0"
    extensions = (".csv", ".tsv")
    task_types = ("binary_classification", "multiclass_classification", "regression")

    @staticmethod
    def inspect(path: Path) -> dict[str, Any]:
        handle, encoding, delimiter = _open_detected(path)
        with handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            columns = list(reader.fieldnames or [])
            samples: list[dict[str, str]] = []
            for index, row in enumerate(reader):
                samples.append({key: str(value or "").strip() for key, value in row.items() if key})
                if index >= 499:
                    break
        numeric = {
            column
            for column in columns
            if samples
            and sum(_number(row.get(column, "")) is not None for row in samples)
            / max(1, sum(bool(row.get(column, "")) for row in samples))
            >= .95
        }
        candidates: list[str] = []
        for column in columns:
            unique = {row.get(column, "") for row in samples if row.get(column, "") != ""}
            if 2 <= len(unique) <= 64:
                candidates.append(column)
            elif column in numeric and len(unique) > 2:
                candidates.append(column)
        preferred = {"target", "label", "class", "y", "answer", "result"}
        suggested = next((column for column in candidates if column.lower() in preferred), None)
        if not suggested and columns and columns[-1] in candidates:
            suggested = columns[-1]
        return {
            "columns": columns,
            "numeric_columns": sorted(numeric),
            "target_candidates": candidates,
            "suggested_target": suggested or (columns[-1] if columns else None),
            "sampled_rows": len(samples),
            "encoding": encoding,
            "delimiter": delimiter,
            "preview": samples[:20],
        }

    @staticmethod
    def import_file(
        path: Path, target: str | None = None
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        handle, encoding, delimiter = _open_detected(path)
        handle.close()
        if encoding in {"utf-8", "utf-8-sig"}:
            try:
                with NativeNumericCsv(path, target, delimiter) as native:
                    x, raw_y = native.arrays()
                    unique = np.unique(raw_y)
                    integer_like = bool(np.equal(unique, np.floor(unique)).all())
                    classification_limit = max(20, int(math.sqrt(len(raw_y))) + 1)
                    if len(unique) == 2:
                        task = TaskType.BINARY_CLASSIFICATION.value
                        y = np.searchsorted(unique, raw_y).astype(np.int64)
                        classes = [str(int(value)) if float(value).is_integer() else str(float(value)) for value in unique]
                        target_dtype = "categorical"
                    elif integer_like and 2 < len(unique) <= classification_limit:
                        task = TaskType.MULTICLASS_CLASSIFICATION.value
                        y = np.searchsorted(unique, raw_y).astype(np.int64)
                        classes = [str(int(value)) for value in unique]
                        target_dtype = "categorical"
                    else:
                        task = TaskType.REGRESSION.value
                        y = raw_y
                        classes = []
                        target_dtype = "number"
                    columns = [
                        {"name": name, "dtype": "number", "nullable": False, "encoded_features": 1}
                        for name in native.feature_names
                    ]
                    transformations = [
                        {
                            "column": name,
                            "kind": "native_standardize",
                            "mean": native.means[index],
                            "std": native.stds[index],
                        }
                        for index, name in enumerate(native.feature_names)
                    ]
                    metadata = {
                        "records": native.rows,
                        "features": native.features,
                        "target": native.target,
                        "classes": classes,
                        "skipped": 0,
                        "task_types": [task],
                        "schema": {
                            "columns": columns,
                            "target": native.target,
                            "target_dtype": target_dtype,
                            "task_types": [task],
                            "feature_count": native.features,
                            "row_count": native.rows,
                        },
                        "transformations": transformations,
                        "warnings": [],
                        "encoding": encoding,
                        "delimiter": delimiter,
                        "engine": "kernelyra-native-csv/1",
                        "validation_report": {
                            "valid": True,
                            "warnings": [],
                            "skipped_rows": 0,
                        },
                    }
                    return metadata, x, y
            except NativeCoreError:
                # Mixed/categorical/missing-value CSV remains supported by the
                # general Python ingestor while numeric data stays on C++.
                pass
        handle, encoding, delimiter = _open_detected(path)
        with handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            field_names: list[str] = list(reader.fieldnames or ())
            prepared = prepare_rows(field_names, reader, target)
        metadata = {
            **prepared.metadata,
            "encoding": encoding,
            "delimiter": delimiter,
            "validation_report": {
                "valid": True,
                "warnings": prepared.metadata["warnings"],
                "skipped_rows": prepared.metadata["skipped"],
            },
        }
        return metadata, prepared.x, prepared.y
