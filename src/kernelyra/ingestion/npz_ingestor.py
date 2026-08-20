from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import DatasetError
from ..models import DatasetSchema
from .tabular import MIN_ROWS, _task_and_target


class NPZIngestor:
    name = "npz"
    version = "1.0"
    extensions = (".npz",)
    task_types = ("binary_classification", "multiclass_classification", "regression")

    @staticmethod
    def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
        try:
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                total_uncompressed = sum(item.file_size for item in entries)
                total_compressed = sum(max(1, item.compress_size) for item in entries)
                if len(entries) > 16 or total_uncompressed > 2 * 1024**3:
                    raise DatasetError("NPZ превышает безопасный лимит распаковки 2 GB или 16 массивов")
                if total_uncompressed / max(1, total_compressed) > 200:
                    raise DatasetError("NPZ отклонён как потенциальная decompression bomb")
                if any(".." in Path(item.filename).parts or Path(item.filename).is_absolute() for item in entries):
                    raise DatasetError("NPZ содержит небезопасные имена записей")
        except zipfile.BadZipFile:
            raise DatasetError("Повреждённый NPZ/ZIP контейнер") from None
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "x" not in archive or "y" not in archive:
                    raise DatasetError("NPZ должен содержать массивы 'x' и 'y'")
                x = np.asarray(archive["x"])
                y = np.asarray(archive["y"])
        except (OSError, ValueError) as error:
            raise DatasetError(f"Небезопасный или повреждённый NPZ: {error}") from None
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            raise DatasetError("NPZ ожидает x[rows, features] и y[rows] одинаковой длины")
        if len(x) < MIN_ROWS or x.shape[1] < 1:
            raise DatasetError(f"NPZ должен содержать минимум {MIN_ROWS} строк и один признак")
        if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            raise DatasetError("Core NPZ ingestor принимает только числовые x и y без pickle")
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise DatasetError("NPZ содержит NaN или бесконечные значения")
        return x, y

    @classmethod
    def inspect(cls, path: Path) -> dict[str, Any]:
        x, y = cls._load(path)
        task, _, classes, _ = _task_and_target([str(value) for value in y])
        return {
            "columns": [f"feature_{index}" for index in range(x.shape[1])] + ["target"],
            "suggested_target": "target",
            "shape": [int(x.shape[0]), int(x.shape[1])],
            "task_types": [task],
            "classes": classes,
            "allow_pickle": False,
        }

    @classmethod
    def import_file(
        cls, path: Path, target: str | None = None
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        if target not in {None, "target", "y"}:
            raise DatasetError("Для NPZ target всегда находится в массиве 'y'")
        x, raw_y = cls._load(path)
        task, y, classes, target_dtype = _task_and_target([str(value) for value in raw_y])
        means = x.mean(axis=0)
        stds = x.std(axis=0)
        keep = stds > 1e-12
        warnings = [
            f"Константный feature_{index} исключён"
            for index, enabled in enumerate(keep)
            if not enabled
        ]
        if not keep.any():
            raise DatasetError("Все NPZ признаки константны")
        x = ((x[:, keep] - means[keep]) / stds[keep]).astype(np.float32)
        columns = tuple(
            {
                "name": f"feature_{index}",
                "dtype": "number",
                "nullable": False,
                "encoded_features": 1,
            }
            for index, enabled in enumerate(keep)
            if enabled
        )
        schema = DatasetSchema(
            columns=columns,
            target="target",
            target_dtype=target_dtype,
            task_types=(task,),
            feature_count=int(x.shape[1]),
            row_count=int(x.shape[0]),
        )
        return {
            "records": int(x.shape[0]),
            "features": int(x.shape[1]),
            "target": "target",
            "classes": classes,
            "skipped": 0,
            "task_types": [task],
            "schema": schema.to_dict(),
            "transformations": [{"kind": "standardize", "source": "x"}],
            "warnings": warnings,
            "validation_report": {"valid": True, "warnings": warnings, "skipped_rows": 0},
            "allow_pickle": False,
        }, x, y
