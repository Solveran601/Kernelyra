from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import queue
import threading
from collections.abc import Generator, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .errors import DatasetError
from .models import TaskType

HASH_BUCKETS = 64
EVALUATION_ROWS = 4096
STREAM_EXTENSIONS = {".csv", ".tsv", ".jsonl", ".ndjson", ".parquet", ".pq"}
MAX_FOLDER_FILES = 100_000


def _split_for_index(index: int) -> str:
    """Stable train/validation/test split with good distribution and no RAM index."""
    value = (index + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    bucket = (value ^ (value >> 31)) % 100
    if bucket < 15:
        return "validation"
    if bucket < 30:
        return "test"
    return "train"


def _number(value: Any) -> float | None:
    text = str(value or "").strip().replace("\u00a0", "")
    if not text:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def _detect_text(path: Path) -> tuple[str, str]:
    sample = path.read_bytes()[:64 * 1024]
    encoding = "utf-8-sig"
    for candidate in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            text = sample.decode(candidate)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    try:
        delimiter = csv.Sniffer().sniff(text, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    return encoding, delimiter


def _iter_csv(path: Path, encoding: str, delimiter: str, engine: str) -> Iterator[dict[str, str]]:
    if engine == "pandas":
        import pandas as pd  # type: ignore[import-untyped]

        with pd.read_csv(
            path,
            sep=delimiter,
            encoding=encoding,
            dtype=str,
            keep_default_na=False,
            chunksize=8192,
        ) as reader:
            for frame in reader:
                for row in frame.to_dict(orient="records"):
                    yield {str(key): str(value or "").strip() for key, value in row.items()}
        return
    with path.open("r", encoding=encoding, newline="") as handle:
        for row in csv.DictReader(handle, delimiter=delimiter):
            yield {str(key): str(value or "").strip() for key, value in row.items() if key}


def _iter_jsonl(path: Path, encoding: str) -> Iterator[dict[str, str]]:
    with path.open("r", encoding=encoding) as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as error:
                raise DatasetError(f"Invalid JSONL at line {line_number}: {error.msg}") from None
            if not isinstance(value, dict):
                raise DatasetError(f"JSONL line {line_number} must contain an object")
            yield {str(key): str(item or "").strip() for key, item in value.items()}


def _iter_parquet(path: Path) -> Iterator[dict[str, str]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        raise DatasetError('Parquet streaming requires: pip install "kernelyra-ai[data]"') from None
    parquet = pq.ParquetFile(path)  # type: ignore[no-untyped-call]
    for batch in parquet.iter_batches(batch_size=8192):  # type: ignore[no-untyped-call]
        columns = batch.to_pydict()
        names = list(columns)
        for index in range(batch.num_rows):
            yield {name: str(columns[name][index] or "").strip() for name in names}


def iter_rows(spec: Mapping[str, Any]) -> Iterator[dict[str, str]]:
    sources = spec.get("sources")
    if isinstance(sources, list):
        for child in sources:
            if not isinstance(child, Mapping):
                raise DatasetError("Folder stream manifest contains an invalid source")
            yield from iter_rows(child)
        return
    path = Path(str(spec["path"]))
    format_name = str(spec["format"])
    if format_name in {"csv", "tsv"}:
        yield from _iter_csv(path, str(spec["encoding"]), str(spec["delimiter"]), str(spec["reader_engine"]))
        return
    if format_name in {"jsonl", "ndjson"}:
        yield from _iter_jsonl(path, str(spec["encoding"]))
        return
    if format_name == "parquet":
        yield from _iter_parquet(path)
        return
    raise DatasetError(f"Streaming format '{format_name}' is not supported")


def _fingerprint(path: Path) -> str:
    if path.is_dir():
        digest = hashlib.sha256(b"kernelyra-folder-stream/1")
        files = _folder_sources(path)
        for item in files:
            stat = item.stat()
            relative = item.relative_to(path).as_posix()
            digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        return digest.hexdigest()
    stat = path.stat()
    digest = hashlib.sha256(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            handle.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(handle.read(1024 * 1024))
    return digest.hexdigest()


def _folder_sources(path: Path) -> list[Path]:
    files: list[Path] = []
    for item in sorted(path.rglob("*"), key=lambda value: value.as_posix().casefold()):
        if item.is_symlink():
            continue
        if item.is_file() and item.suffix.lower() in STREAM_EXTENSIONS:
            files.append(item)
            if len(files) > MAX_FOLDER_FILES:
                raise DatasetError(f"Dataset folder exceeds the {MAX_FOLDER_FILES} file safety limit")
    if not files:
        raise DatasetError("Dataset folder has no streamable CSV/TSV/JSONL/NDJSON/Parquet files")
    return files


def _source_reader(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower().lstrip(".")
    format_name = "jsonl" if suffix == "ndjson" else "parquet" if suffix == "pq" else suffix
    if format_name not in {"csv", "tsv", "jsonl", "parquet"}:
        raise DatasetError("External streaming currently supports CSV, TSV, JSONL/NDJSON and Parquet")
    encoding, delimiter = ("binary", "") if format_name == "parquet" else _detect_text(path)
    reader_engine = "pandas" if format_name in {"csv", "tsv"} and importlib.util.find_spec("pandas") else "stdlib"
    return {
        "contract": "kernelyra-stream-source/1",
        "path": str(path),
        "format": format_name,
        "encoding": encoding,
        "delimiter": delimiter,
        "reader_engine": reader_engine,
    }


def _native_numeric_spec(
    source: Path,
    target: str | None,
    format_name: str,
    encoding: str,
    delimiter: str,
) -> dict[str, Any] | None:
    if format_name not in {"csv", "tsv"} or encoding not in {"utf-8", "utf-8-sig"}:
        return None
    try:
        from .native_core import NativeCoreError, NativeNumericCsvScan

        with NativeNumericCsvScan(source, target, delimiter) as scan:
            if scan.split_records["validation"] < 8 or scan.split_records["test"] < 8:
                return None
            numeric_values = [float(value) for value in scan.target_values]
            if len(set(numeric_values)) != len(numeric_values):
                return None
            integer_like = all(value.is_integer() for value in numeric_values)
            limit = max(20, int(math.sqrt(scan.rows)) + 1)
            if scan.target_values_overflow:
                task = TaskType.REGRESSION.value
            elif len(numeric_values) == 2:
                task = TaskType.BINARY_CLASSIFICATION.value
            elif 2 < len(numeric_values) <= min(64, limit) and integer_like:
                task = TaskType.MULTICLASS_CLASSIFICATION.value
            else:
                task = TaskType.REGRESSION.value
            return {
                "contract": "kernelyra-stream/1",
                "path": str(source),
                "format": format_name,
                "encoding": encoding,
                "delimiter": delimiter,
                "reader_engine": "kernelyra-native-csv-scan/1",
                "target": scan.target,
                "columns": list(scan.columns),
                "feature_columns": list(scan.feature_names),
                "numeric_columns": list(scan.feature_names),
                "categorical_columns": [],
                "means": scan.means,
                "stds": scan.stds,
                "hash_buckets": HASH_BUCKETS,
                "task_type": task,
                "classes": list(scan.target_values) if task != TaskType.REGRESSION.value else [],
                "records": scan.rows,
                "split_records": scan.split_records,
                "features": len(scan.feature_names),
                "size_bytes": source.stat().st_size,
                "fingerprint": _fingerprint(source),
                "fingerprint_kind": "size-mtime-first-last-1m-sha256",
            }
    except (NativeCoreError, UnicodeError, ValueError):
        return None


def build_stream_spec(path: str | Path, target: str | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        files = _folder_sources(source)
        sources = [_source_reader(item) for item in files]
        base = {
            "contract": "kernelyra-stream/2",
            "path": str(source),
            "format": "folder",
            "reader_engine": "multi-source",
            "sources": sources,
            "file_count": len(files),
        }
        size_bytes = sum(item.stat().st_size for item in files)
    elif source.is_file():
        base = _source_reader(source)
        base["contract"] = "kernelyra-stream/2"
        format_name = str(base["format"])
        native_spec = _native_numeric_spec(
            source,
            target,
            format_name,
            str(base["encoding"]),
            str(base["delimiter"]),
        )
        if native_spec is not None:
            return native_spec
        size_bytes = source.stat().st_size
    else:
        raise DatasetError("Dataset file or folder was not found")
    rows = iter_rows(base)
    sample: list[dict[str, str]] = []
    try:
        for _ in range(1000):
            sample.append(next(rows))
    except StopIteration:
        pass
    if len(sample) < 32:
        raise DatasetError("Streaming dataset requires at least 32 valid rows")
    columns = list(sample[0])
    if any(list(row) != columns for row in sample):
        raise DatasetError("Dataset rows do not have a stable column schema")
    selected_target = target or next(
        (column for column in columns if column.lower() in {"target", "label", "class", "y", "answer", "result"}),
        columns[-1] if columns else None,
    )
    if not selected_target or selected_target not in columns:
        raise DatasetError("Target column was not found")
    feature_columns = [column for column in columns if column != selected_target]
    if not feature_columns:
        raise DatasetError("Dataset has no feature columns")
    numeric_columns = {
        column
        for column in feature_columns
        if sum(_number(row.get(column)) is not None for row in sample) / len(sample) >= .95
    }
    sums = {column: 0.0 for column in numeric_columns}
    sums_sq = {column: 0.0 for column in numeric_columns}
    numeric_counts = {column: 0 for column in numeric_columns}
    target_numeric = True
    target_unique: set[str] = set()
    row_count = 0
    split_records = {"train": 0, "validation": 0, "test": 0}

    def consume(row: Mapping[str, str]) -> None:
        nonlocal row_count, target_numeric
        if list(row) != columns:
            raise DatasetError(f"Folder dataset schema changed near global row {row_count + 1}")
        split = _split_for_index(row_count)
        split_records[split] += 1
        row_count += 1
        target_value = str(row.get(selected_target, "")).strip()
        if not target_value:
            raise DatasetError(f"Target contains a missing value near row {row_count}")
        if _number(target_value) is None:
            target_numeric = False
        if len(target_unique) <= 65:
            target_unique.add(target_value)
        if split != "train":
            return
        for column in numeric_columns:
            value = _number(row.get(column))
            if value is None:
                continue
            sums[column] += value
            sums_sq[column] += value * value
            numeric_counts[column] += 1

    for row in sample:
        consume(row)
    for row in rows:
        consume(row)
    if len(target_unique) == 2:
        task = TaskType.BINARY_CLASSIFICATION.value
    elif not target_numeric:
        if len(target_unique) > 64:
            raise DatasetError("Categorical target has more than 64 classes")
        task = TaskType.MULTICLASS_CLASSIFICATION.value
    else:
        numeric_unique = sorted(
            value for raw in target_unique if (value := _number(raw)) is not None
        )
        integer_like = all(float(value).is_integer() for value in numeric_unique)
        limit = max(20, int(math.sqrt(row_count)) + 1)
        task = (
            TaskType.MULTICLASS_CLASSIFICATION.value
            if len(numeric_unique) <= min(64, limit) and integer_like
            else TaskType.REGRESSION.value
        )
    classes = sorted(target_unique) if task != TaskType.REGRESSION.value else []
    if task != TaskType.REGRESSION.value and len(classes) > 64:
        raise DatasetError("Target has more than 64 classes")
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for column in numeric_columns:
        count = max(1, numeric_counts[column])
        mean = sums[column] / count
        variance = max(0.0, sums_sq[column] / count - mean * mean)
        means[column] = mean
        stds[column] = math.sqrt(variance) or 1.0
    categorical = [column for column in feature_columns if column not in numeric_columns]
    feature_count = len(numeric_columns) + len(categorical) * HASH_BUCKETS
    if feature_count > 200_000:
        raise DatasetError("Encoded feature space is too wide for the core streaming pipeline")
    return {
        **base,
        "target": selected_target,
        "columns": columns,
        "feature_columns": feature_columns,
        "numeric_columns": sorted(numeric_columns),
        "categorical_columns": categorical,
        "means": means,
        "stds": stds,
        "hash_buckets": HASH_BUCKETS,
        "task_type": task,
        "classes": classes,
        "records": row_count,
        "split_records": split_records,
        "features": feature_count,
        "size_bytes": size_bytes,
        "fingerprint": _fingerprint(source),
        "fingerprint_kind": "size-mtime-first-last-1m-sha256",
    }


class StreamingTabularSource:
    def __init__(self, spec: Mapping[str, Any], seed: int, data_workers: int = 0, prefetch: int = 0):
        self.spec = dict(spec)
        self.seed = seed
        source = Path(str(self.spec["path"]))
        if not source.exists() or _fingerprint(source) != self.spec.get("fingerprint"):
            raise DatasetError("Streaming dataset changed after it was registered")
        self.data_workers = max(0, int(data_workers))
        self.prefetch = max(0, int(prefetch))
        self.epoch = 0
        self.rows_consumed = 0
        self._iterator = self._training_rows()
        self._executor = (
            ThreadPoolExecutor(max_workers=self.data_workers, thread_name_prefix="kernelyra-data")
            if self.data_workers > 1
            else None
        )
        self._queue: queue.Queue[Any] | None = None
        self._producer: threading.Thread | None = None
        self._stop = threading.Event()
        self._closed = False
        self.validation_x, self.validation_y, self.test_x, self.test_y = self._evaluation_arrays()

    @property
    def train_records(self) -> int:
        split = self.spec.get("split_records") or {}
        return max(1, int(split.get("train") or int(int(self.spec["records"]) * .70)))

    def _split(self, index: int) -> str:
        return _split_for_index(index)

    def _encode(self, row: Mapping[str, str]) -> tuple[np.ndarray, float]:
        result: list[float] = []
        means = self.spec["means"]
        stds = self.spec["stds"]
        for column in self.spec["numeric_columns"]:
            value = _number(row.get(column))
            result.append(0.0 if value is None else (value - float(means[column])) / float(stds[column]))
        buckets = int(self.spec["hash_buckets"])
        for column in self.spec["categorical_columns"]:
            encoded = [0.0] * buckets
            raw = str(row.get(column, ""))
            bucket = int.from_bytes(hashlib.blake2b(raw.encode("utf-8"), digest_size=8).digest(), "little") % buckets
            encoded[bucket] = 1.0
            result.extend(encoded)
        target = str(row[self.spec["target"]]).strip()
        if self.spec["task_type"] == TaskType.REGRESSION.value:
            number = _number(target)
            if number is None:
                raise DatasetError("Regression target contains a non-numeric value")
            y = number
        else:
            try:
                y = float(self.spec["classes"].index(target))
            except ValueError:
                raise DatasetError("Target class changed after streaming inspection") from None
        return np.asarray(result, dtype=np.float32), y

    def _training_rows(self) -> Generator[dict[str, str], None, None]:
        while True:
            emitted = 0
            for index, row in enumerate(iter_rows(self.spec)):
                if self._split(index) != "train":
                    continue
                emitted += 1
                yield row
            if emitted == 0:
                raise DatasetError("Streaming dataset produced no training rows")

    def _evaluation_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        validation: list[tuple[np.ndarray, float]] = []
        test: list[tuple[np.ndarray, float]] = []
        for index, row in enumerate(iter_rows(self.spec)):
            split = self._split(index)
            if split == "validation" and len(validation) < EVALUATION_ROWS:
                validation.append(self._encode(row))
            elif split == "test" and len(test) < EVALUATION_ROWS:
                test.append(self._encode(row))
            if len(validation) >= EVALUATION_ROWS and len(test) >= EVALUATION_ROWS:
                break
        if len(validation) < 8 or len(test) < 8:
            raise DatasetError("Streaming dataset produced insufficient validation/test rows")

        def arrays(values: list[tuple[np.ndarray, float]]) -> tuple[np.ndarray, np.ndarray]:
            return np.stack([item[0] for item in values]), np.asarray([item[1] for item in values])

        validation_x, validation_y = arrays(validation)
        test_x, test_y = arrays(test)
        return validation_x, validation_y, test_x, test_y

    def _encode_many(self, rows: list[dict[str, str]]) -> list[tuple[np.ndarray, float]]:
        if self._executor is None:
            return [self._encode(row) for row in rows]
        return list(self._executor.map(self._encode, rows))

    def _next_values(self, count: int) -> list[tuple[np.ndarray, float]]:
        rows = [next(self._iterator) for _ in range(count)]
        return self._encode_many(rows)

    def _producer_loop(self, chunk_size: int) -> None:
        queue_target = self._queue
        if queue_target is None:
            raise RuntimeError("Streaming queue was not initialized")
        try:
            while not self._stop.is_set():
                for item in self._next_values(chunk_size):
                    while not self._stop.is_set():
                        try:
                            queue_target.put(item, timeout=.1)
                            break
                        except queue.Full:
                            continue
        except BaseException as error:
            while not self._stop.is_set():
                try:
                    queue_target.put(error, timeout=.1)
                    break
                except queue.Full:
                    continue

    def _ensure_prefetch(self, batch_size: int) -> None:
        if self.prefetch <= 0 or self._producer is not None:
            return
        capacity = max(batch_size, batch_size * self.prefetch)
        self._queue = queue.Queue(maxsize=capacity)
        self._producer = threading.Thread(
            target=self._producer_loop,
            args=(max(1, min(batch_size, 256)),),
            name="kernelyra-prefetch",
            daemon=True,
        )
        self._producer.start()

    def next_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        if self._closed:
            raise DatasetError("Streaming source is closed")
        if batch_size < 1:
            raise DatasetError("Batch size must be positive")
        self._ensure_prefetch(batch_size)
        if self._queue is None:
            values = self._next_values(batch_size)
        else:
            values = []
            for _ in range(batch_size):
                item = self._queue.get()
                if isinstance(item, BaseException):
                    self._stop.set()
                    raise item
                values.append(item)
        self.rows_consumed += len(values)
        self.epoch = self.rows_consumed // self.train_records
        return np.stack([item[0] for item in values]), np.asarray([item[1] for item in values])

    def state(self) -> dict[str, int]:
        return {"stream_epoch": self.epoch, "stream_rows_consumed": self.rows_consumed}

    def _stop_producer(self) -> None:
        if self._producer is None:
            return
        self._stop.set()
        self._producer.join(timeout=5)
        if self._producer.is_alive():
            raise DatasetError("Streaming prefetch worker did not stop safely")
        self._producer = None
        self._queue = None
        self._stop = threading.Event()

    def restore_rows(self, rows_consumed: int) -> None:
        if rows_consumed < 0:
            raise DatasetError("Streaming checkpoint row count is invalid")
        self._stop_producer()
        self._iterator.close()
        self._iterator = self._training_rows()
        offset = rows_consumed % self.train_records
        for _ in range(offset):
            next(self._iterator)
        self.rows_consumed = rows_consumed
        self.epoch = rows_consumed // self.train_records

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_producer()
        self._iterator.close()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
