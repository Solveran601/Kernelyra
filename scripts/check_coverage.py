from __future__ import annotations

import json
from pathlib import Path

from coverage import Coverage

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "kernelyra"

# Core means dataset preparation, the built-in NumPy engine, persistence,
# scheduling/lifecycle, policy/security and the public in-process workspace API.
# Transport adapters, optional engines, CLI/API and platform subprocess
# glue have separate integration, conformance and Windows gates.
CORE_FILES = (
    "agent_policy.py",
    "backends/base.py",
    "backends/numpy_backend.py",
    "batch.py",
    "checkpoints.py",
    "datasets.py",
    "ingestion/csv_ingestor.py",
    "ingestion/jsonl_ingestor.py",
    "ingestion/npz_ingestor.py",
    "ingestion/registry.py",
    "ingestion/router.py",
    "ingestion/tabular.py",
    "metrics.py",
    "models.py",
    "runtime.py",
    "security.py",
    "storage/sqlite.py",
    "workspace.py",
)
CRITICAL_FILES = ("runtime.py", "security.py")


def line_coverage(cov: Coverage, files: tuple[str, ...]) -> dict[str, object]:
    statements = 0
    missing = 0
    details: list[dict[str, object]] = []
    for relative in files:
        path = SOURCE / relative
        _, executable, _, uncovered, _ = cov.analysis2(str(path))
        file_statements = len(executable)
        file_missing = len(uncovered)
        file_percent = 100.0 if not file_statements else 100.0 * (file_statements - file_missing) / file_statements
        statements += file_statements
        missing += file_missing
        details.append(
            {
                "file": f"src/kernelyra/{relative}",
                "statements": file_statements,
                "missing": file_missing,
                "line_percent": round(file_percent, 2),
            }
        )
    percent = 100.0 if not statements else 100.0 * (statements - missing) / statements
    return {
        "statements": statements,
        "missing": missing,
        "line_percent": round(percent, 2),
        "files": details,
    }


def main() -> int:
    cov = Coverage(data_file=str(ROOT / ".coverage"))
    cov.load()
    core = line_coverage(cov, CORE_FILES)
    critical = line_coverage(cov, CRITICAL_FILES)
    result = {
        "contract": "kernelyra-coverage-gate/1",
        "core": {**core, "required_line_percent": 85.0},
        "security_lifecycle_state_machine": {**critical, "required_line_percent": 90.0},
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if float(core["line_percent"]) < 85.0:
        raise SystemExit("Kernelyra core line coverage is below 85%")
    if float(critical["line_percent"]) < 90.0:
        raise SystemExit("Security/lifecycle/state-machine line coverage is below 90%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
