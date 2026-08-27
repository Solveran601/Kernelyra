"""Bounded data-quality diagnostics and reproducible dataset contracts.

The doctor deliberately works from the router's bounded preview.  Its findings
are preflight signals, never claims about rows that were not inspected.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .planning import ContextChunkPlanner

CONTRACT_VERSION = "kernelyra-dataset-contract/1"
_TARGET_HINTS = {"target", "label", "class", "y", "answer", "result"}
_TIME_HINT = re.compile(r"(?:^|[_\-])(date|time|timestamp|created|updated|event)(?:$|[_\-])", re.I)
_GROUP_HINT = re.compile(r"(?:^|[_\-])(group|user|account|customer|session|conversation|document|device)(?:$|[_\-])", re.I)
_ID_HINT = re.compile(r"(?:^|[_\-])(id|uuid|guid|key)(?:$|[_\-])", re.I)
_MAX_COLUMNS = 512
_MAX_CLASSES = 64
_TEXT_TABLE_SUFFIXES = {".csv", ".tsv", ".jsonl", ".ndjson"}
_RECORD_ESTIMATE_BYTES = 8 * 1024 * 1024


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _number(value: Any) -> float | None:
    text = _text(value).replace("\u00a0", "")
    if not text:
        return None
    try:
        parsed = float(text.replace(",", "."))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _rows(inspection: Mapping[str, Any]) -> list[dict[str, Any]]:
    preview = inspection.get("preview")
    if not isinstance(preview, Sequence) or isinstance(preview, str | bytes):
        return []
    return [dict(row) for row in preview if isinstance(row, Mapping)]


def estimate_records(path: str | Path, inspection: Mapping[str, Any]) -> int:
    """Estimate records through one bounded read using the same rule as planning.

    Routers deliberately return only a small preview.  The doctor and the
    planner must nevertheless agree on the dataset scale because the result
    controls batching and the variable context chunks.  For newline-delimited
    tables this function reads at most 8 MiB; all other formats keep the
    preview-based estimate supplied by their router.
    """
    source = Path(path).expanduser().resolve()
    shape = inspection.get("shape") or []
    known = inspection.get("rows") or (shape[0] if len(shape) >= 2 else None)
    if known:
        return max(1, int(known))

    sampled = max(1, int(inspection.get("sampled_rows") or 1))
    preview = _rows(inspection)
    preview_bytes = max(1, sum(len(str(row)) for row in preview))
    size = int(inspection.get("bytes") or (source.stat().st_size if source.exists() else 0))
    fallback = max(sampled, int(size / max(1, preview_bytes / max(1, len(preview)))))
    if not source.is_file() or source.suffix.lower() not in _TEXT_TABLE_SUFFIXES:
        return fallback

    sample_size = min(size, _RECORD_ESTIMATE_BYTES)
    try:
        with source.open("rb") as handle:
            prefix = handle.read(sample_size)
    except OSError:
        return fallback

    lines = prefix.count(b"\n")
    if source.suffix.lower() in {".csv", ".tsv"}:
        lines = max(0, lines - 1)
    if size <= sample_size:
        return max(sampled, lines + (1 if prefix and not prefix.endswith(b"\n") else 0))
    bytes_per_record = sample_size / max(1, lines)
    return max(sampled, int(size / bytes_per_record))


def _issue(code: str, severity: str, message: str, **evidence: Any) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "evidence": evidence}


def _task_from_inspection(inspection: Mapping[str, Any], target_values: list[str]) -> str | None:
    tasks = [str(item) for item in inspection.get("task_types") or []]
    if len(tasks) == 1:
        return tasks[0]
    unique = set(target_values)
    if len(unique) == 2:
        return "binary_classification"
    if not unique:
        return None
    if all(_number(item) is not None for item in unique) and len(unique) > max(20, int(len(target_values) ** .5) + 1):
        return "regression"
    return "multiclass_classification"


def _split_policy(columns: list[str], task: str | None, *, target: str | None, row_count: int) -> dict[str, Any]:
    candidates = [column for column in columns if column != target]
    temporal = next((column for column in candidates if _TIME_HINT.search(column)), None)
    context = next((column for column in candidates if _GROUP_HINT.search(column)), None)
    if temporal:
        return {
            "strategy": "temporal",
            "execution_strategy": "temporal",
            "order_column": temporal,
            "context_column": None,
            "reason": "A time-like column was found; preserve source order to reduce future-to-past leakage.",
            "enforcement": "materialized backends preserve source order; the input is not re-sorted by this column",
        }
    if context:
        return {
            "strategy": "context",
            "execution_strategy": "context",
            "order_column": None,
            "context_column": context,
            "reason": "A group-like column was found; keep every matching context in exactly one split.",
            "enforcement": "external streaming tabular path; AutoTrainer routes detected context data through the built-in group-exclusive splitter",
        }
    if task in {"binary_classification", "multiclass_classification"}:
        return {
            "strategy": "stratified",
            "execution_strategy": "stratified",
            "order_column": None,
            "context_column": None,
            "reason": "Classification labels are partitioned deterministically by class.",
            "enforcement": "materialized backends",
        }
    return {
            "strategy": "random",
            "execution_strategy": "random",
        "order_column": None,
        "context_column": None,
        "reason": "No temporal, context, or classification signal was available.",
        "enforcement": "materialized backends",
    }


def recommend_chunk_policy(
    records: int,
    features: int,
    *,
    seed: int = 42,
    target_records: int | None = None,
) -> dict[str, Any]:
    """Return a variable, contiguous chunk plan without materialising dataset rows."""
    records = max(0, int(records))
    features = max(1, int(features))
    automatic_target = max(512, min(16_384, 2 ** max(9, min(14, int(math.log2(max(2, 1_048_576 // features)))))))
    target = automatic_target if target_records is None else max(128, min(262_144, int(target_records)))
    planner = ContextChunkPlanner(target_records=target, seed=seed)
    summary = planner.summary(records, preview=6)
    return {
        "strategy": "adaptive_contiguous_ranges",
        "reason": "Variable ranges reduce synchronized allocation spikes while preserving input order.",
        "target_records": target,
        "seed": int(seed),
        **summary,
    }


def analyze_inspection(
    inspection: Mapping[str, Any],
    *,
    target: str | None = None,
    records_estimate: int | None = None,
    feature_count: int | None = None,
    seed: int = 42,
    chunk_target_records: int | None = None,
) -> dict[str, Any]:
    """Analyze one bounded router inspection and return JSON-safe diagnostics."""
    all_columns = [str(column) for column in inspection.get("columns") or [] if _text(column)]
    columns = all_columns[:_MAX_COLUMNS]
    sampled = _rows(inspection)
    selected_target = target or inspection.get("suggested_target")
    selected_target = str(selected_target) if selected_target in all_columns else None
    findings: list[dict[str, Any]] = []
    if len(all_columns) > len(columns):
        findings.append(
            _issue(
                "schema_preview_truncated",
                "info",
                "Only the first 512 columns are included in the bounded preflight report.",
                total_columns=len(all_columns),
            )
        )
    if not sampled:
        findings.append(
            _issue(
                "preview_unavailable",
                "warning",
                "The format was recognized but no bounded row preview is available for data-health checks.",
            )
        )

    column_stats: list[dict[str, Any]] = []
    target_values: list[str] = []
    for column in columns:
        values = [_text(row.get(column)) for row in sampled]
        present = [value for value in values if value]
        unique = set(present)
        numeric_count = sum(_number(value) is not None for value in present)
        numeric_ratio = numeric_count / max(1, len(present))
        stats = {
            "name": column,
            "sampled": len(values),
            "non_empty": len(present),
            "missing_rate": round(1.0 - len(present) / max(1, len(values)), 6),
            "unique_values": len(unique),
            "unique_rate": round(len(unique) / max(1, len(present)), 6),
            "inferred_type": "numeric" if numeric_ratio >= 0.95 and present else "categorical",
        }
        column_stats.append(stats)
        if column == selected_target:
            target_values = present
        if stats["missing_rate"] >= 0.05:
            findings.append(
                _issue(
                    "missing_values",
                    "warning",
                    "Sampled values contain missing entries; verify the preprocessing policy before training.",
                    column=column,
                    missing_rate=stats["missing_rate"],
                )
            )
        named_identifier = bool(_ID_HINT.search(column))
        high_cardinality_category = (
            stats["inferred_type"] == "categorical"
            and stats["unique_rate"] >= 0.98
            and stats["unique_values"] >= 20
        )
        if column != selected_target and (named_identifier or high_cardinality_category):
            findings.append(
                _issue(
                    "identifier_like_feature",
                    "warning",
                    "This feature looks identifier-like in the sample and can overfit or leak identity.",
                    column=column,
                    unique_rate=stats["unique_rate"],
                    inferred_type=stats["inferred_type"],
                    basis="column_name" if named_identifier else "high_cardinality_categorical",
                )
            )

    canonical_rows = [json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in sampled]
    duplicate_count = len(canonical_rows) - len(set(canonical_rows))
    if duplicate_count:
        findings.append(
            _issue(
                "duplicate_rows",
                "warning",
                "The bounded preview contains duplicate rows.",
                duplicates=duplicate_count,
                sampled_rows=len(sampled),
            )
        )
    task = _task_from_inspection(inspection, target_values)
    distribution: dict[str, int] = {}
    if selected_target and target_values and task != "regression":
        counts = Counter(target_values)
        distribution = dict(counts.most_common(_MAX_CLASSES))
        largest = max(counts.values()) / len(target_values)
        if len(counts) > _MAX_CLASSES:
            findings.append(
                _issue(
                    "many_target_classes",
                    "warning",
                    "The sampled target has more than 64 classes; confirm that this is intentional.",
                    classes=len(counts),
                )
            )
        if len(counts) >= 2 and largest >= 0.8:
            findings.append(
                _issue(
                    "class_imbalance",
                    "warning",
                    "One class dominates the sampled target; inspect minority-class recall, not only accuracy.",
                    largest_class_fraction=round(largest, 6),
                )
            )
        for column in columns:
            if column == selected_target:
                continue
            pairs = [(_text(row.get(column)), _text(row.get(selected_target))) for row in sampled]
            comparable = [(left, right) for left, right in pairs if left and right]
            if len(comparable) >= 8:
                equal_rate = sum(left == right for left, right in comparable) / len(comparable)
                if equal_rate >= 0.98:
                    findings.append(
                        _issue(
                            "possible_target_leakage",
                            "error",
                            "A feature matches the target in nearly every sampled row; exclude or justify it before training.",
                            column=column,
                            equal_rate=round(equal_rate, 6),
                        )
                    )

    estimated_records = int(records_estimate or inspection.get("rows") or inspection.get("sampled_rows") or len(sampled))
    estimated_features = int(feature_count or max(1, len(all_columns) - (1 if selected_target else 0)))
    split_policy = _split_policy(all_columns, task, target=selected_target, row_count=estimated_records)
    chunk_policy = recommend_chunk_policy(
        estimated_records,
        estimated_features,
        seed=seed,
        target_records=chunk_target_records,
    )
    warnings = [item["message"] for item in findings if item["severity"] in {"warning", "error"}]
    source = {
        "path": str(inspection.get("path") or ""),
        "format": str(inspection.get("format") or "unknown"),
        "size_bytes": int(inspection.get("bytes") or 0),
    }
    contract = {
        "contract": CONTRACT_VERSION,
        "source": source,
        "schema": {
            "columns": all_columns,
            "target": selected_target,
            "task": task,
            "sampled_rows": len(sampled),
            "estimated_records": estimated_records,
            "estimated_features": estimated_features,
        },
        "split_policy": split_policy,
        "chunk_policy": chunk_policy,
    }
    encoded = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    contract["signature"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return {
        "contract": contract,
        "summary": {
            "sampled_rows": len(sampled),
            "columns": len(all_columns),
            "target": selected_target,
            "task": task,
            "duplicate_rows": duplicate_count,
            "findings": len(findings),
            "blocking_findings": sum(item["severity"] == "error" for item in findings),
        },
        "columns": column_stats,
        "target_distribution": distribution,
        "findings": findings,
        "warnings": warnings,
    }


def inspect_path(path: str | Path, router: Any, *, target: str | None = None, seed: int = 42) -> dict[str, Any]:
    """Inspect a local path through the existing router, then add data-health evidence."""
    source = Path(path).expanduser().resolve()
    inspection = router.inspect(source)
    return {
        "inspection": inspection,
        **analyze_inspection(
            inspection,
            target=target,
            records_estimate=estimate_records(source, inspection),
            seed=seed,
        ),
    }
