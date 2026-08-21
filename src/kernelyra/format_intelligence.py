"""Truthful, bounded format inspection for the V2 data intake path.

Recognition is not support.  This module combines the explicit format
catalogue with a small Rust signature probe and recommends only an action the
installed build can honestly perform.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .formats import FORMAT_COUNT, FormatDescriptor, format_for_path
from .native_core import NativeCore, NativeCoreError

_SIGNATURE_NAMES = {
    0: "unknown",
    1: "parquet",
    2: "sqlite",
    3: "numpy",
    4: "hdf5",
    5: "zip-container",
    6: "gzip",
    7: "pdf",
    8: "png",
    9: "jpeg",
    10: "gif",
    11: "riff",
    12: "flac",
    13: "ogg",
    14: "iso-base-media",
    15: "matroska",
    16: "json",
    17: "delimited-text",
}

_EXPECTED_SIGNATURES = {
    "structured-text": {16, 17},
    "columnar-array": {1, 3, 4},
    "database": {2},
    "office-document": {5, 7},
    "spreadsheet": {5},
    "raster-image": {8, 9, 10, 11},
    "vector-image": {5, 7},
    "audio": {11, 12, 13},
    "video": {14, 15},
    "archive": {5, 6},
}

_DIRECT_FORMATS = {"csv", "tsv", "jsonl", "ndjson", "parquet", "pq", "npz"}
_SAMPLE_BYTES = 4096


def _rust_probe(path: Path) -> tuple[int | None, str]:
    with path.open("rb") as handle:
        prefix = handle.read(_SAMPLE_BYTES)
    try:
        code = NativeCore().probe_signature(prefix)
    except (NativeCoreError, OSError):
        return None, "native-core-unavailable"
    if code is None:
        return None, "native-core-v1"
    return code, "rust-signature"


def _recommendation(descriptor: FormatDescriptor | None) -> dict[str, Any]:
    if descriptor is None:
        return {
            "strategy": "manual-classification-required",
            "direct_training": False,
            "next_action": "Choose a trusted adapter or convert the data to CSV, JSONL, NPZ or Parquet.",
        }
    if descriptor.id in _DIRECT_FORMATS:
        return {
            "strategy": "direct-tabular-training",
            "direct_training": True,
            "next_action": "Inspect the schema, choose a target column, then use plan or train.",
            "preferred_backend": "native when available; NumPy remains the fallback",
        }
    if descriptor.training == "extract":
        dependency = f" Install the optional {descriptor.dependency} adapter." if descriptor.dependency else ""
        return {
            "strategy": "extract-then-tabularize",
            "direct_training": False,
            "next_action": "Extract features into a tabular dataset before training." + dependency,
        }
    if descriptor.role == "model":
        return {
            "strategy": "inspect-only",
            "direct_training": False,
            "next_action": "Do not execute or deserialize an untrusted model container; inspect it in an isolated workflow.",
        }
    return {
        "strategy": "adapter-required",
        "direct_training": False,
        "next_action": "This format is catalogued but has no built-in trainer in this release.",
    }


def advise_path(raw_path: str | Path) -> dict[str, Any]:
    """Inspect a local file using at most 4 KiB and return a safe AI data plan."""
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("Format advice requires an existing regular file")
    descriptor = format_for_path(path)
    signature_code, engine = _rust_probe(path)
    expected = _EXPECTED_SIGNATURES.get(descriptor.category) if descriptor else None
    signature_matches: bool | None
    if signature_code is None or signature_code == 0 or not expected:
        signature_matches = None
    else:
        signature_matches = signature_code in expected
    warnings: list[str] = []
    if descriptor is None:
        warnings.append("The extension is outside the built-in catalogue; no training route is assumed.")
    if signature_matches is False:
        warnings.append("The file signature conflicts with its extension; inspect it before extraction or training.")
    if signature_code is None:
        warnings.append("The bundled native core predates the V2 Rust signature probe; only catalogue evidence is available.")
    return {
        "format_intelligence": "v2-preview",
        "path": str(path),
        "bytes": path.stat().st_size,
        "catalogue_routes": FORMAT_COUNT,
        "catalogue": descriptor.to_dict() if descriptor else None,
        "signature": {
            "engine": engine,
            "sample_bytes": min(path.stat().st_size, _SAMPLE_BYTES),
            "code": signature_code,
            "detected": _SIGNATURE_NAMES.get(signature_code or 0, "unknown"),
            "matches_expected_family": signature_matches,
        },
        "recommendation": _recommendation(descriptor),
        "warnings": warnings,
        "contract": "catalogued != signature-verified != extractable != directly-trainable",
    }
