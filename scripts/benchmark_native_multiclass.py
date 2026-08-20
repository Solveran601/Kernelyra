"""Compare one-thread and OpenMP multiclass native training on identical data."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from kernelyra.native_core import NativeModel


def run(x: np.ndarray, y: np.ndarray, classes: int, steps: int, threads: int) -> tuple[float, float]:
    with NativeModel(
        task="multiclass_classification", features=x.shape[1], classes=classes,
        seed=20260820, learning_rate=.01, threads=threads,
    ) as model:
        started = time.perf_counter()
        for _ in range(steps):
            model.train_step(x, y)
        seconds = time.perf_counter() - started
        accuracy = float((model.predict(x).argmax(axis=1) == y).mean())
    return seconds, accuracy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    rng = np.random.default_rng(20260820)
    x = rng.normal(size=(args.rows, args.features)).astype(np.float32)
    teacher = rng.normal(size=(args.features, args.classes)).astype(np.float32)
    y = (x @ teacher).argmax(axis=1).astype(np.float32)
    sequential = [run(x, y, args.classes, args.steps, 1) for _ in range(args.runs)]
    parallel = [run(x, y, args.classes, args.steps, args.threads) for _ in range(args.runs)]
    sequential_seconds = statistics.median(value[0] for value in sequential)
    parallel_seconds = statistics.median(value[0] for value in parallel)
    payload = {
        "workload": vars(args),
        "one_thread": {"seconds": sequential_seconds, "updates_per_second": args.steps / sequential_seconds, "accuracy": statistics.median(value[1] for value in sequential)},
        "openmp": {"seconds": parallel_seconds, "updates_per_second": args.steps / parallel_seconds, "accuracy": statistics.median(value[1] for value in parallel)},
        "speedup_openmp_over_one_thread": sequential_seconds / parallel_seconds,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
