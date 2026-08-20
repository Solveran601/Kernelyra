from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import DatasetError
from ..models import DatasetSchema, TaskType

MAX_CATEGORIES = 64
MIN_ROWS = 40


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    metadata: dict[str, Any]
    x: np.ndarray
    y: np.ndarray


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _task_and_target(values: list[str]) -> tuple[str, np.ndarray, list[str], str]:
    numeric = [_number(value) for value in values]
    all_numeric = all(value is not None for value in numeric)
    if all_numeric:
        numbers = np.asarray(numeric, dtype=np.float64)
        unique = sorted(set(float(value) for value in numbers))
        integer_like = all(float(value).is_integer() for value in unique)
        classification_limit = max(20, int(math.sqrt(len(numbers))) + 1)
        if len(unique) == 2:
            labels = [str(int(value)) if value.is_integer() else str(value) for value in unique]
            encoded = np.asarray([unique.index(float(value)) for value in numbers], dtype=np.int64)
            return TaskType.BINARY_CLASSIFICATION.value, encoded, labels, "categorical"
        if integer_like and 2 < len(unique) <= classification_limit:
            labels = [str(int(value)) for value in unique]
            encoded = np.asarray([unique.index(float(value)) for value in numbers], dtype=np.int64)
            return TaskType.MULTICLASS_CLASSIFICATION.value, encoded, labels, "categorical"
        return TaskType.REGRESSION.value, numbers.astype(np.float64), [], "number"

    classes = sorted(set(values))
    if len(classes) < 2:
        raise DatasetError("Target должен содержать хотя бы два различных значения")
    if len(classes) > MAX_CATEGORIES:
        raise DatasetError(
            f"Target содержит {len(classes)} категорий; максимум для core pipeline — {MAX_CATEGORIES}"
        )
    task = (
        TaskType.BINARY_CLASSIFICATION.value
        if len(classes) == 2
        else TaskType.MULTICLASS_CLASSIFICATION.value
    )
    return task, np.asarray([classes.index(value) for value in values], dtype=np.int64), classes, "categorical"


def prepare_rows(
    columns: list[str],
    rows: Iterable[dict[str, Any]],
    target: str | None,
) -> PreparedDataset:
    if len(columns) < 2:
        raise DatasetError("Dataset должен содержать хотя бы один признак и target")
    target = target or columns[-1]
    if target not in columns:
        raise DatasetError(f"Столбец target '{target}' не найден")

    materialized: list[dict[str, str]] = []
    skipped = 0
    for raw in rows:
        normalized = {column: str(raw.get(column, "") if raw.get(column) is not None else "").strip() for column in columns}
        if not normalized[target]:
            skipped += 1
            continue
        materialized.append(normalized)
    if len(materialized) < MIN_ROWS:
        raise DatasetError(f"После проверки осталось меньше {MIN_ROWS} строк с непустым target")

    target_values = [row[target] for row in materialized]
    task, y, classes, target_dtype = _task_and_target(target_values)
    feature_arrays: list[np.ndarray] = []
    feature_schema: list[dict[str, Any]] = []
    transformations: list[dict[str, Any]] = []
    warnings: list[str] = []

    for column in (item for item in columns if item != target):
        raw_values = [row[column] for row in materialized]
        nonempty = [value for value in raw_values if value]
        if not nonempty:
            warnings.append(f"Пустой столбец '{column}' исключён")
            continue
        if all(value == target_value for value, target_value in zip(raw_values, target_values, strict=False)):
            warnings.append(f"Столбец '{column}' исключён как точная утечка target")
            continue

        parsed = [_number(value) for value in raw_values]
        numeric_ratio = sum(value is not None for value in parsed) / max(1, len(nonempty))
        if numeric_ratio >= .95:
            present = np.asarray([value for value in parsed if value is not None], dtype=np.float64)
            median = float(np.median(present))
            numeric_values = np.asarray([median if value is None else value for value in parsed], dtype=np.float64)
            mean = float(numeric_values.mean())
            std = float(numeric_values.std())
            if std <= 1e-12:
                warnings.append(f"Константный столбец '{column}' исключён")
                continue
            feature_arrays.append(((numeric_values - mean) / std)[:, None])
            missing = sum(value is None for value in parsed)
            feature_schema.append(
                {"name": column, "dtype": "number", "nullable": missing > 0, "encoded_features": 1}
            )
            transformations.append(
                {
                    "column": column,
                    "kind": "median_impute_standardize",
                    "median": median,
                    "mean": mean,
                    "std": std,
                }
            )
            continue

        categorical_values = [value if value else "<missing>" for value in raw_values]
        categories = sorted(set(categorical_values))
        if len(categories) == 1:
            warnings.append(f"Константный столбец '{column}' исключён")
            continue
        if len(categories) > MAX_CATEGORIES:
            raise DatasetError(
                f"Категориальный столбец '{column}' содержит {len(categories)} значений; "
                f"максимум — {MAX_CATEGORIES}"
            )
        encoded = np.zeros((len(categorical_values), len(categories)), dtype=np.float64)
        lookup = {value: index for index, value in enumerate(categories)}
        for row_index, value in enumerate(categorical_values):
            encoded[row_index, lookup[value]] = 1.0
        feature_arrays.append(encoded)
        feature_schema.append(
            {
                "name": column,
                "dtype": "category",
                "nullable": "<missing>" in categories,
                "categories": categories,
                "encoded_features": len(categories),
            }
        )
        transformations.append({"column": column, "kind": "one_hot", "categories": categories})

    if not feature_arrays:
        raise DatasetError("После проверки не осталось пригодных признаков")
    x = np.concatenate(feature_arrays, axis=1).astype(np.float32, copy=False)
    if not np.isfinite(x).all() or not np.isfinite(y.astype(np.float64)).all():
        raise DatasetError("Dataset содержит NaN или бесконечные значения после преобразования")

    schema = DatasetSchema(
        columns=tuple(feature_schema),
        target=target,
        target_dtype=target_dtype,
        task_types=(task,),
        feature_count=int(x.shape[1]),
        row_count=int(x.shape[0]),
    )
    return PreparedDataset(
        metadata={
            "records": int(x.shape[0]),
            "features": int(x.shape[1]),
            "target": target,
            "classes": classes,
            "skipped": skipped,
            "task_types": [task],
            "schema": schema.to_dict(),
            "transformations": transformations,
            "warnings": warnings,
        },
        x=x,
        y=y,
    )
