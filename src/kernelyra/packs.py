"""Validated, user-editable algorithm packs.

Built-in packs are immutable release defaults.  A user pack is a small JSON
record that selects a built-in base, enables performance algorithms and may
override bounded tuning values.  Correctness guards are deliberately not pack
options: dataset boundaries, allocation limits and Model Guard always remain
active.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PACK_ALGORITHMS: dict[str, dict[str, str]] = {
    "bulk_training_dispatch": {
        "area": "training",
        "effect": "Groups repeated native training steps into one ABI call.",
    },
    "thread_parallel_gradient": {
        "area": "training",
        "effect": "Allows the native gradient engine to use the configured CPU threads.",
    },
    "parallel_data_prefetch": {
        "area": "input",
        "effect": "Enables bounded worker prefetch for upcoming dataset chunks.",
    },
    "wide_context_chunks": {
        "area": "input",
        "effect": "Allows larger context-preserving chunks from the base policy.",
    },
    "expanded_tensor_arena": {
        "area": "memory",
        "effect": "Allows a reusable native tensor arena above the balanced ceiling.",
    },
}


BUILTIN_ALGORITHM_PACKS: dict[str, dict[str, Any]] = {
    "careful": {
        "label": "Careful stream",
        "profile": "low-memory",
        "algorithms": ("bulk_training_dispatch",),
        "data_workers": 0,
        "prefetch": 1,
        "stream_limit": 128 * 1024 * 1024,
        "native_thread_fraction": .35,
        "bulk_step_cap": 8,
        "arena_bytes": 32 * 1024 * 1024,
        "chunk_target_records": 1024,
        "hidden_layers": (32, 16),
        "strategy": "small variable contiguous chunks with conservative memory reuse",
        "cpu_backends": ("native", "numpy", "torch", "tensorflow"),
        "hybrid_backends": ("torch", "tensorflow", "native", "numpy"),
    },
    "balanced": {
        "label": "Balanced stream",
        "profile": "balanced",
        "algorithms": (
            "bulk_training_dispatch",
            "thread_parallel_gradient",
            "parallel_data_prefetch",
        ),
        "data_workers": 2,
        "prefetch": 2,
        "stream_limit": 256 * 1024 * 1024,
        "native_thread_fraction": .60,
        "bulk_step_cap": 32,
        "arena_bytes": 96 * 1024 * 1024,
        "chunk_target_records": 4096,
        "hidden_layers": (64, 32),
        "strategy": "variable chunks and bounded parallel prefetch",
        "cpu_backends": ("native", "numpy", "torch", "tensorflow"),
        "hybrid_backends": ("torch", "tensorflow", "native", "numpy"),
    },
    "throughput": {
        "label": "Throughput stream",
        "profile": "performance",
        "algorithms": tuple(PACK_ALGORITHMS),
        "data_workers": 6,
        "prefetch": 4,
        "stream_limit": 384 * 1024 * 1024,
        "native_thread_fraction": .85,
        "bulk_step_cap": 100,
        "arena_bytes": 256 * 1024 * 1024,
        "chunk_target_records": 16384,
        "hidden_layers": (128, 64, 32),
        "strategy": "larger variable chunks with parallel prefetch and bulk dispatch",
        "cpu_backends": ("native", "numpy", "torch", "tensorflow"),
        "hybrid_backends": ("torch", "tensorflow", "native", "numpy"),
    },
    "maximum": {
        "label": "Maximum local throughput",
        "profile": "workstation",
        "algorithms": tuple(PACK_ALGORITHMS),
        "data_workers": 12,
        "prefetch": 8,
        "stream_limit": 2**63 - 1,
        "native_thread_fraction": 1.0,
        "bulk_step_cap": 100,
        "arena_bytes": 768 * 1024 * 1024,
        "chunk_target_records": 65536,
        "hidden_layers": (256, 128, 64),
        "strategy": "large variable chunks and maximum local parallel dispatch within explicit limits",
        "cpu_backends": ("native", "numpy", "torch", "tensorflow"),
        "hybrid_backends": ("torch", "tensorflow", "native", "numpy"),
    },
}

_PACK_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_OVERRIDE_RULES: dict[str, tuple[type, float, float]] = {
    "data_workers": (int, 0, 64),
    "prefetch": (int, 1, 64),
    "stream_limit": (int, 1024**2, 2**63 - 1),
    "native_thread_fraction": (float, .05, 1.0),
    "bulk_step_cap": (int, 1, 10000),
    "arena_bytes": (int, 8 * 1024**2, 8 * 1024**3),
    "chunk_target_records": (int, 32, 1_000_000),
}


def algorithm_pack_path() -> Path:
    """Return the JSON table used for custom packs."""
    configured = os.environ.get("KERNELYRA_PACKS_FILE", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "Kernelyra" / "packs.json"
    return Path.home() / ".kernelyra" / "packs.json"


def _normalise_name(value: str) -> str:
    name = str(value).strip().lower()
    if not _PACK_NAME.fullmatch(name):
        raise ValueError("pack name must match [a-z0-9][a-z0-9._-]{0,63}")
    return name


def _normalise_algorithms(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list | tuple):
        raise ValueError("algorithms must be a JSON array")
    result: list[str] = []
    for value in values:
        name = str(value).strip().lower()
        if name not in PACK_ALGORITHMS:
            raise ValueError(f"unknown pack algorithm: {value}")
        if name not in result:
            result.append(name)
    return tuple(result)


def _normalise_overrides(values: Any) -> dict[str, int | float]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError("overrides must be a JSON object")
    result: dict[str, int | float] = {}
    for key, value in values.items():
        if key not in _OVERRIDE_RULES:
            raise ValueError(f"unsupported pack override: {key}")
        expected, minimum, maximum = _OVERRIDE_RULES[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"pack override {key} must be numeric")
        converted: int | float = int(value) if expected is int else float(value)
        if not minimum <= converted <= maximum:
            raise ValueError(f"pack override {key} must be between {minimum:g} and {maximum:g}")
        result[key] = converted
    return result


def _read_registry() -> dict[str, dict[str, Any]]:
    path = algorithm_pack_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read algorithm pack table {path}: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("schema") != 1 or not isinstance(payload.get("packs"), Mapping):
        raise ValueError(f"invalid algorithm pack table schema in {path}")
    result: dict[str, dict[str, Any]] = {}
    for raw_name, raw_record in payload["packs"].items():
        name = _normalise_name(str(raw_name))
        if name in BUILTIN_ALGORITHM_PACKS:
            raise ValueError(f"custom table cannot replace built-in pack: {name}")
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"custom pack {name} must be a JSON object")
        base = _normalise_name(str(raw_record.get("base", "balanced")))
        if base not in BUILTIN_ALGORITHM_PACKS:
            raise ValueError(f"custom pack {name} has unknown built-in base: {base}")
        result[name] = {
            "base": base,
            "label": str(raw_record.get("label") or name)[:120],
            "algorithms": _normalise_algorithms(raw_record.get("algorithms", [])),
            "overrides": _normalise_overrides(raw_record.get("overrides", {})),
        }
    return result


def _write_registry(records: Mapping[str, Mapping[str, Any]]) -> Path:
    path = algorithm_pack_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "packs": {
            name: {
                "base": record["base"],
                "label": record["label"],
                "algorithms": list(record["algorithms"]),
                "overrides": dict(record["overrides"]),
            }
            for name, record in sorted(records.items())
        },
    }
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(pending, path)
    return path


def resolve_pack_name(value: str | None) -> str:
    name = _normalise_name(value or "balanced")
    if name in BUILTIN_ALGORITHM_PACKS or name in _read_registry():
        return name
    raise KeyError(f"Unknown algorithm pack: {value}")


def _apply_algorithm_effects(policy: dict[str, Any]) -> dict[str, Any]:
    """Materialize algorithm switches into values consumed by every frontend."""
    algorithms = set(policy["algorithms"])
    if "bulk_training_dispatch" not in algorithms:
        policy["bulk_step_cap"] = 1
    if "parallel_data_prefetch" not in algorithms:
        policy["data_workers"] = 0
        policy["prefetch"] = 1
    if "wide_context_chunks" not in algorithms:
        policy["chunk_target_records"] = min(4096, int(policy["chunk_target_records"]))
    if "expanded_tensor_arena" not in algorithms:
        policy["arena_bytes"] = min(96 * 1024**2, int(policy["arena_bytes"]))
    return policy


def get_algorithm_pack(name: str) -> dict[str, Any]:
    selected = _normalise_name(name)
    if selected in BUILTIN_ALGORITHM_PACKS:
        policy = dict(BUILTIN_ALGORITHM_PACKS[selected])
        policy.update({"name": selected, "base": selected, "built_in": True})
        return _apply_algorithm_effects(policy)
    records = _read_registry()
    if selected not in records:
        raise KeyError(f"Unknown algorithm pack: {name}")
    record = records[selected]
    policy = dict(BUILTIN_ALGORITHM_PACKS[record["base"]])
    policy.update(record["overrides"])
    policy.update(
        {
            "name": selected,
            "base": record["base"],
            "label": record["label"],
            "algorithms": tuple(record["algorithms"]),
            "built_in": False,
        }
    )
    return _apply_algorithm_effects(policy)


def list_algorithm_packs() -> dict[str, dict[str, Any]]:
    names = [*BUILTIN_ALGORITHM_PACKS, *_read_registry()]
    return {name: get_algorithm_pack(name) for name in names}


def algorithm_pack_table() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "base": pack["base"],
            "label": pack["label"],
            "built_in": pack["built_in"],
            "algorithms": list(pack["algorithms"]),
        }
        for name, pack in list_algorithm_packs().items()
    ]


def create_algorithm_pack(
    name: str,
    *,
    base: str = "balanced",
    label: str | None = None,
    overrides: Mapping[str, int | float] | None = None,
) -> dict[str, Any]:
    selected = _normalise_name(name)
    if selected in BUILTIN_ALGORITHM_PACKS:
        raise ValueError(f"built-in pack is immutable: {selected}")
    source = get_algorithm_pack(resolve_pack_name(base))
    records = _read_registry()
    if selected in records:
        raise ValueError(f"algorithm pack already exists: {selected}")
    records[selected] = {
        "base": source["base"],
        "label": (label or selected)[:120],
        "algorithms": tuple(source["algorithms"]),
        "overrides": _normalise_overrides(overrides),
    }
    _write_registry(records)
    return get_algorithm_pack(selected)


def add_pack_algorithm(pack: str, algorithm: str) -> dict[str, Any]:
    selected = _normalise_name(pack)
    algorithm_name = _normalise_algorithms([algorithm])[0]
    records = _read_registry()
    if selected not in records:
        if selected in BUILTIN_ALGORITHM_PACKS:
            raise ValueError("built-in packs are immutable; clone one before editing it")
        raise KeyError(f"Unknown algorithm pack: {pack}")
    algorithms = list(records[selected]["algorithms"])
    if algorithm_name not in algorithms:
        algorithms.append(algorithm_name)
    records[selected]["algorithms"] = tuple(algorithms)
    _write_registry(records)
    return get_algorithm_pack(selected)


def remove_pack_algorithm(pack: str, algorithm: str) -> dict[str, Any]:
    selected = _normalise_name(pack)
    algorithm_name = _normalise_algorithms([algorithm])[0]
    records = _read_registry()
    if selected not in records:
        if selected in BUILTIN_ALGORITHM_PACKS:
            raise ValueError("built-in packs are immutable; clone one before editing it")
        raise KeyError(f"Unknown algorithm pack: {pack}")
    records[selected]["algorithms"] = tuple(
        item for item in records[selected]["algorithms"] if item != algorithm_name
    )
    _write_registry(records)
    return get_algorithm_pack(selected)


def delete_algorithm_pack(pack: str) -> Path:
    selected = _normalise_name(pack)
    if selected in BUILTIN_ALGORITHM_PACKS:
        raise ValueError(f"built-in pack is immutable: {selected}")
    records = _read_registry()
    if selected not in records:
        raise KeyError(f"Unknown algorithm pack: {pack}")
    del records[selected]
    return _write_registry(records)
