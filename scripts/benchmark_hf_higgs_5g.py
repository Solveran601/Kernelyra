"""Reproducible 4.94 GiB Hugging Face streaming benchmark for Kernelyra.

The selected public data is ``yzhuang/autotree_automl_Higgs_gosdt_l256_d3_sd0``.
Its fifteen Parquet shards total about 4.94 GiB.  This script separates three
claims that are too often conflated:

* download integrity and sustained storage throughput;
* Kernelyra native policy throughput versus a pure-Python policy; and
* structural Parquet verification (requires ``kernelyra-ai[parquet]``).

It does *not* claim to train an LLM on Parquet bytes.  The dataset stores nested
AutoML traces, so model-quality validation belongs to the flat labelled HIGGS
example in ``examples/guarded_higgs_training.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from kernelyra.native_core import NativeCore, native_core_status


REPOSITORY = "yzhuang/autotree_automl_Higgs_gosdt_l256_d3_sd0"
REVISION = "6d627a98e326eed1b0e1a089e50d83d01de2f8c2"
BASE_URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}"
TRAIN_SHARDS = [
    "train-00000-of-00013-32888e9d7b8a067f.parquet",
    "train-00001-of-00013-89a7a97eb6f2a965.parquet",
    "train-00002-of-00013-f763aba0d9f79a1e.parquet",
    "train-00003-of-00013-674f5e61b0d4ca7d.parquet",
    "train-00004-of-00013-a2ff8d3b0b8823f5.parquet",
    "train-00005-of-00013-cf86598b3e4f848c.parquet",
    "train-00006-of-00013-0e9c76eb306ff385.parquet",
    "train-00007-of-00013-098481303a2a3605.parquet",
    "train-00008-of-00013-64fb4611a0cddc88.parquet",
    "train-00009-of-00013-8e59dbc6055a7475.parquet",
    "train-00010-of-00013-c821454bb571def4.parquet",
    "train-00011-of-00013-353c82ef4b139d15.parquet",
    "train-00012-of-00013-70a5de55b0c08676.parquet",
]
VALIDATION_SHARDS = [
    "validation-00000-of-00002-b671bf454cfe9532.parquet",
    "validation-00001-of-00002-4dc0c4fa40cedb92.parquet",
]
ALL_SHARDS = TRAIN_SHARDS + VALIDATION_SHARDS


@dataclass(frozen=True)
class Result:
    name: str
    bytes_read: int
    seconds: float
    mib_per_second: float
    digest: str


def local_paths(root: Path) -> list[Path]:
    return [root / "data" / name for name in ALL_SHARDS]


def download(root: Path) -> None:
    for path in local_paths(root):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size:
            print(f"reuse {path.name} ({path.stat().st_size} bytes)")
            continue
        url = f"{BASE_URL}/data/{path.name}"
        partial = path.with_suffix(path.suffix + ".partial")
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        print(f"download {path.name} from byte {offset}")
        with urllib.request.urlopen(request, timeout=60) as source:
            resumed = offset and getattr(source, "status", None) == 206
            with partial.open("ab" if resumed else "wb") as destination:
                while chunk := source.read(8 * 1024 * 1024):
                    destination.write(chunk)
        partial.replace(path)


def _scan_fixed(paths: Iterable[Path], chunk_size: int) -> Result:
    digest = hashlib.sha256()
    total = 0
    started = time.perf_counter()
    for path in paths:
        with path.open("rb", buffering=chunk_size) as handle:
            while block := handle.read(chunk_size):
                total += len(block)
                digest.update(block)
    seconds = time.perf_counter() - started
    return Result("python_fixed_4MiB", total, seconds, total / 2**20 / seconds, digest.hexdigest())


def _scan_native_policy(paths: Iterable[Path], native: NativeCore) -> Result:
    """Read identical bytes while asking the Rust policy for bounded chunks."""
    digest = hashlib.sha256()
    total = 0
    ordinal = 0
    started = time.perf_counter()
    for path in paths:
        remaining = path.stat().st_size
        with path.open("rb", buffering=4 * 1024 * 1024) as handle:
            while remaining:
                chunk_size = native.next_chunk_size(
                    remaining,
                    4 * 1024 * 1024,
                    3 * 1024 * 1024,
                    5 * 1024 * 1024,
                    ordinal,
                    20260820,
                )
                block = handle.read(chunk_size)
                if not block:
                    break
                total += len(block)
                remaining -= len(block)
                ordinal += 1
                digest.update(block)
    seconds = time.perf_counter() - started
    return Result("kernelyra_rust_variable_policy", total, seconds, total / 2**20 / seconds, digest.hexdigest())


def parquet_structure(paths: Iterable[Path]) -> dict[str, object]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        return {"available": False, "reason": f"{error}; install kernelyra-ai[parquet]"}
    rows = 0
    row_groups = 0
    columns: list[str] | None = None
    for path in paths:
        file = parquet.ParquetFile(path)
        rows += file.metadata.num_rows
        row_groups += file.metadata.num_row_groups
        columns = columns or list(file.schema.names)
    return {"available": True, "rows": rows, "row_groups": row_groups, "columns": columns}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(".benchmarks/hf-higgs-5g"))
    parser.add_argument("--download", action="store_true", help="Fetch missing public shards (~4.94 GiB)")
    parser.add_argument("--allow-partial", action="store_true", help="Benchmark only completed shards; report is marked incomplete")
    parser.add_argument("--report", type=Path, default=Path(".benchmarks/hf-higgs-5g/report.json"))
    args = parser.parse_args()
    paths = local_paths(args.root)
    if args.download:
        download(args.root)
    missing = [path for path in paths if not path.is_file()]
    if missing and not args.allow_partial:
        raise SystemExit("missing shards; rerun with --download: " + ", ".join(str(path) for path in missing[:3]))
    paths = [path for path in paths if path.is_file()]
    if not paths:
        raise SystemExit("no completed shards available")

    native = NativeCore()
    baseline = _scan_fixed(paths, 4 * 1024 * 1024)
    accelerated = _scan_native_policy(paths, native)
    if baseline.digest != accelerated.digest:
        raise RuntimeError("benchmark integrity failure: byte digests differ")
    payload = {
        "dataset": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "files": len(paths),
            "complete": not missing,
            "missing_files": [path.name for path in missing],
        },
        "native": native_core_status(),
        "benchmarks": [asdict(baseline), asdict(accelerated)],
        "speed_ratio_native_over_fixed": accelerated.mib_per_second / baseline.mib_per_second,
        "parquet_structure": parquet_structure(paths),
        "interpretation": "This is an I/O plus chunk-policy benchmark, not an LLM training claim.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
