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


def run_cycle(iteration: int) -> dict[str, object]:
    code = f"""
from pathlib import Path
import time
from kernelyra import RunConfig, Workspace
root = Path({str(ROOT / 'build' / 'soak-runtime').__repr__()}) / 'cycle-{iteration}'
workspace = Workspace.open(root)
run = workspace.create_run(RunConfig(dataset='demo', backend='numpy', target_metric=.5, max_steps=100)).start()
deadline = time.monotonic() + 30
while run.status not in {{'completed','error_recoverable'}} and time.monotonic() < deadline:
    time.sleep(.05)
    run = workspace.runs.get(run.id).info
assert run.status == 'completed', run.to_dict()
assert 'test' in run.metrics, run.metrics
assert workspace.close()
"""
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src")}
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=ROOT,
        env=environment,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=45,
    )
    return {
        "iteration": iteration,
        "ok": result.returncode == 0,
        "seconds": time.monotonic() - started,
        "error": result.stderr[-1000:] if result.returncode else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Kernelyra lifecycle and failure soak")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--duration-minutes", type=float)
    args = parser.parse_args()
    duration = args.duration_minutes if args.duration_minutes is not None else (1.0 if args.mode == "quick" else 60.0)
    minimum_cycles = 2 if args.mode == "quick" else 100
    maximum_cycles = 3 if args.mode == "quick" else 1000
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src")}
    suite_started = time.monotonic()
    critical = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q", *CRITICAL_SUITES],
        cwd=ROOT,
        env=environment,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=max(300, int(duration * 60) + 180),
    )
    results: list[dict[str, object]] = []
    deadline = suite_started + max(1.0, duration * 60)
    iteration = 0
    while iteration < minimum_cycles or (time.monotonic() < deadline and iteration < maximum_cycles):
        outcome = run_cycle(iteration)
        results.append(outcome)
        iteration += 1
        if not outcome["ok"]:
            break
    report = {
        "contract": "kernelyra-soak/1",
        "mode": args.mode,
        "requested_minutes": duration,
        "actual_seconds": time.monotonic() - suite_started,
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "critical_suite_ok": critical.returncode == 0,
        "critical_suite_tail": (critical.stdout + critical.stderr)[-2000:],
        "cycles": results,
        "ok": critical.returncode == 0 and bool(results) and all(bool(item["ok"]) for item in results),
        "scenarios": [
            "create/start/complete",
            "pause/resume/stop/worker.close",
            "daemon restart and duplicate daemon lock",
            "worker crash containment",
            "corrupted checkpoint",
            "stale PID and scheduler lock",
            "concurrent CLI clients and SQLite contention",
            "MCP stdio, agent expiration/revocation and approval replay",
            "closed built-in backend and format registries",
            "Unicode and space paths",
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
