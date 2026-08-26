"""Measure V3 preflight and planning on a synthetic tabular CSV.

This is a local workflow benchmark, not a competitor comparison. It records
input size, machine information and measured elapsed time in JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from kernelyra import Engine  # noqa: E402


def make_csv(path: Path, rows: int, features: int) -> None:
    fields = [f"feature_{index}" for index in range(features)] + ["label"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in range(rows):
            values = {f"feature_{index}": round(((row + 3) * (index + 5) % 101) / 100, 4) for index in range(features)}
            values["label"] = int(sum(float(values[f"feature_{index}"]) for index in range(min(3, features))) >= 1.3)
            writer.writerow(values)


def measure(action: object, repeats: int) -> dict[str, float]:
    durations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        action()  # type: ignore[operator]
        durations.append(time.perf_counter() - started)
    return {
        "runs": float(repeats),
        "min_seconds": min(durations),
        "median_seconds": sorted(durations)[len(durations) // 2],
        "max_seconds": max(durations),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / ".benchmarks" / "v3-workflow.json")
    args = parser.parse_args()
    if args.rows < 64 or args.features < 1 or args.repeats < 1:
        raise SystemExit("rows must be >= 64, features and repeats must be positive")

    with tempfile.TemporaryDirectory(prefix="kernelyra-v3-benchmark-") as temporary:
        workspace = Path(temporary)
        dataset = workspace / "benchmark.csv"
        make_csv(dataset, args.rows, args.features)
        with Engine(workspace) as engine:
            doctor = engine.doctor(dataset, "label")
            plan = engine.plan(dataset, "label", backend="numpy", profile="eco", max_steps=16)
            result = {
                "contract": "kernelyra-v3-workflow-benchmark/1",
                "kind": "local_preflight_and_plan",
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "kernelyra": __import__("kernelyra").__version__,
                },
                "input": {
                    "rows": args.rows,
                    "features": args.features,
                    "bytes": dataset.stat().st_size,
                    "repeats": args.repeats,
                },
                "policies": {
                    "bounded_doctor": {
                        "sampled_rows": doctor["summary"]["sampled_rows"],
                        "estimated_records": doctor["contract"]["schema"]["estimated_records"],
                    },
                    "resolved_plan": {
                        "records_estimate": plan.records_estimate,
                        "split": plan.split_policy,
                        "chunk": plan.chunk_policy,
                    },
                },
                "measurements": {
                    "data_doctor": measure(lambda: engine.doctor(dataset, "label"), args.repeats),
                    "plan_numpy": measure(
                        lambda: engine.plan(dataset, "label", backend="numpy", profile="eco", max_steps=16),
                        args.repeats,
                    ),
                },
                "limitations": [
                    "Measures only local V3 preflight and planning, not model quality or framework superiority.",
                    "Elapsed time varies with storage, CPU load, Python and dependency versions.",
                ],
            }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
