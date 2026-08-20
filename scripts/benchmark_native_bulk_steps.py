"""Measure native training dispatch overhead without changing the optimizer path."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from kernelyra.native_core import NativeModel


def measure(*, bulk: bool, x: np.ndarray, y: np.ndarray, batch: int, steps: int, seed: int) -> tuple[float, float]:
    with NativeModel(
        task="binary_classification", features=x.shape[1], seed=seed, learning_rate=.03, threads=1
    ) as model:
        started = time.perf_counter()
        if bulk:
            loss = model.train_random_steps(x, y, batch, steps)
        else:
            for _ in range(steps):
                loss = model.train_random_step(x, y, batch)
        elapsed = time.perf_counter() - started
        accuracy = float(((model.predict(x) >= .5) == y).mean())
    return elapsed, accuracy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--features", type=int, default=28)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    x = rng.normal(size=(args.rows, args.features)).astype(np.float32)
    y = (x[:, 0] - .7 * x[:, 1] + .2 * x[:, 2] > 0).astype(np.float32)
    individual = [measure(bulk=False, x=x, y=y, batch=args.batch, steps=args.steps, seed=args.seed) for _ in range(args.runs)]
    bulk = [measure(bulk=True, x=x, y=y, batch=args.batch, steps=args.steps, seed=args.seed) for _ in range(args.runs)]
    individual_seconds = statistics.median(item[0] for item in individual)
    bulk_seconds = statistics.median(item[0] for item in bulk)
    individual_accuracy = statistics.median(item[1] for item in individual)
    bulk_accuracy = statistics.median(item[1] for item in bulk)
    payload = {
        "workload": {"rows": args.rows, "features": args.features, "batch": args.batch, "steps": args.steps, "runs": args.runs},
        "individual": {"seconds": individual_seconds, "updates_per_second": args.steps / individual_seconds, "accuracy": individual_accuracy},
        "bulk": {"seconds": bulk_seconds, "updates_per_second": args.steps / bulk_seconds, "accuracy": bulk_accuracy},
        "speedup_bulk_over_individual": individual_seconds / bulk_seconds,
        "accuracy_gap_bulk_minus_individual": bulk_accuracy - individual_accuracy,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
