"""Reproducible one-thread dense binary training benchmark.

The process pins common BLAS/OpenMP runtimes to one thread before importing
NumPy. Kernelyra and NumPy start from identical zero parameters, process the
same float32 matrix, use the same update equations and report the last-step
loss plus final accuracy. Runs are interleaved to reduce ordering and thermal
bias on laptops.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402 - thread limits must be set before NumPy loads BLAS

from kernelyra.native_core import NativeCore, NativeModel  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True, help="Path to kernelyra_core.dll")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON evidence")
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--learning-rate", type=float, default=.05)
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--host-label", default="unspecified")
    parser.add_argument("--cpu-label", default=platform.processor() or "unspecified")
    return parser.parse_args()


def timed_native(
    core_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    threads: int,
) -> dict[str, float]:
    core = NativeCore(core_path)
    model = NativeModel(
        task="binary_classification",
        features=x.shape[1],
        threads=threads,
        seed=7,
        learning_rate=learning_rate,
        core=core,
    )
    model.import_parameters(initial_weights, initial_bias)
    started = time.perf_counter_ns()
    loss = model.train_steps(x, y, steps)
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
    probabilities = model.predict(x)
    accuracy = float(np.mean((probabilities >= .5) == y))
    model.close()
    return {"seconds": elapsed, "loss": float(loss), "accuracy": accuracy}


def timed_numpy(
    x: np.ndarray,
    y: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, float]:
    weights = initial_weights.copy()
    bias = np.float32(initial_bias[0])
    loss = 0.0
    started = time.perf_counter_ns()
    for step in range(steps):
        scores = x @ weights + bias
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
        errors = probabilities - y
        weights -= learning_rate * (x.T @ errors / len(y))
        bias -= learning_rate * np.mean(errors)
        if step + 1 == steps:
            bounded = np.clip(probabilities, 1.0e-7, 1.0 - 1.0e-7)
            loss = float(-np.mean(y * np.log(bounded) + (1.0 - y) * np.log(1.0 - bounded)))
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(x @ weights + bias, -30.0, 30.0)))
    accuracy = float(np.mean((probabilities >= .5) == y))
    return {"seconds": elapsed, "loss": loss, "accuracy": accuracy}


def runtime_description() -> str:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        np.show_runtime()
    value = output.getvalue().strip()
    home = str(Path.home())
    value = value.replace(home, "<user-home>")
    value = value.replace(home.replace("\\", "\\\\"), "<user-home>")
    node = platform.node()
    return value.replace(node, "<host>") if node else value


def summarize(samples: list[dict[str, float]], steps: int) -> dict[str, Any]:
    seconds = [sample["seconds"] for sample in samples]
    median_seconds = statistics.median(seconds)
    return {
        "median_seconds": median_seconds,
        "steps_per_second": steps / median_seconds,
        "loss": samples[-1]["loss"],
        "accuracy": samples[-1]["accuracy"],
        "runs_seconds": seconds,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(f".{destination.name}.pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(pending, destination)


def main() -> int:
    args = arguments()
    if min(args.rows, args.features, args.steps, args.repeats, args.warmups, args.native_threads) < 1:
        raise SystemExit("rows, features, steps, repeats, warmups and native threads must be positive")
    core_path = args.core.expanduser().resolve()
    rng = np.random.default_rng(args.seed)
    x = np.ascontiguousarray(rng.normal(size=(args.rows, args.features)), dtype=np.float32)
    truth = np.ascontiguousarray(rng.normal(size=args.features), dtype=np.float32)
    y = np.ascontiguousarray((x @ truth > 0).astype(np.float32))
    initial_weights = np.zeros(args.features, dtype=np.float32)
    initial_bias = np.zeros(1, dtype=np.float32)
    core = NativeCore(core_path)

    runners: dict[str, Callable[[], dict[str, float]]] = {
        "kernelyra_native": lambda: timed_native(
            core_path,
            x,
            y,
            initial_weights,
            initial_bias,
            steps=args.steps,
            learning_rate=args.learning_rate,
            threads=args.native_threads,
        ),
        "numpy": lambda: timed_numpy(
            x,
            y,
            initial_weights,
            initial_bias,
            steps=args.steps,
            learning_rate=args.learning_rate,
        ),
    }
    for _ in range(args.warmups):
        for runner in runners.values():
            runner()
    samples = {name: [] for name in runners}
    names = list(runners)
    for repeat in range(args.repeats):
        order = names[repeat % len(names):] + names[: repeat % len(names)]
        for name in order:
            samples[name].append(runners[name]())

    results = {name: summarize(values, args.steps) for name, values in samples.items()}
    native_seconds = results["kernelyra_native"]["median_seconds"]
    numpy_seconds = results["numpy"]["median_seconds"]
    payload = {
        "schema": "kernelyra-cpu-benchmark/1",
        "claim_scope": "One local dense binary float32 workload; not a universal framework comparison.",
        "host": {
            "label": args.host_label,
            "cpu": args.cpu_label,
            "logical_threads": os.cpu_count(),
            "platform": platform.platform(),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "numpy_runtime": runtime_description(),
            "native_version": core.version,
            "native_features": core.features,
            "native_components": core.components,
            "thread_environment": {
                name: os.environ[name]
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "workload": {
            "rows": args.rows,
            "features": args.features,
            "steps": args.steps,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "dtype": "float32",
            "native_threads": args.native_threads,
            "parameter_initialization": "identical zeros",
            "measurement_order": "alternating",
            "loss_reporting": "last bulk step only for both implementations",
        },
        "results": results,
        "comparison": {
            "native_speedup_vs_numpy": numpy_seconds / native_seconds,
            "native_time_over_numpy": native_seconds / numpy_seconds,
            "winner_for_this_case": "kernelyra_native" if native_seconds < numpy_seconds else "numpy",
            "accuracy_delta": (
                results["kernelyra_native"]["accuracy"] - results["numpy"]["accuracy"]
            ),
            "absolute_loss_delta": abs(
                results["kernelyra_native"]["loss"] - results["numpy"]["loss"]
            ),
        },
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
