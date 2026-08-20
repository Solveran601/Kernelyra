from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRITICAL_SUITES = (
    "tests/test_runtime.py",
    "tests/test_runtime_contracts.py",
    "tests/test_runtime_shutdown.py",
    "tests/test_state_machine.py",
    "tests/test_process_worker.py",
    "tests/test_security.py",
    "tests/test_mcp_stdio.py",
    "tests/test_daemon_integration.py",
    "tests/test_failure_modes.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Kernelyra critical contracts sequentially")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    count = max(1, min(100, args.count))
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src")}
    results: list[dict[str, object]] = []
    for iteration in range(1, count + 1):
        started = time.monotonic()
        completed = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "-q", *CRITICAL_SUITES],
            cwd=ROOT,
            env=environment,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=300,
        )
        results.append(
            {
                "iteration": iteration,
                "ok": completed.returncode == 0,
                "seconds": round(time.monotonic() - started, 3),
                "tail": (completed.stdout + completed.stderr)[-1000:],
            }
        )
        if completed.returncode:
            break
    report = {
        "contract": "kernelyra-critical-repeat/1",
        "requested": count,
        "completed": len(results),
        "ok": len(results) == count and all(bool(item["ok"]) for item in results),
        "results": results,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
