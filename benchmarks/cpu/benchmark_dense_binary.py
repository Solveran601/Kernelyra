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


def native_pipeline_probe(
    core_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float,
) -> dict[str, Any]:
    """Prove engine participation outside the timed full-batch workload.

    The measured workload intentionally uses full batches.  The native random
    batch route exercises the Rust sampler and Zig gather path, while its
    small dense update chooses the portable Fortran arithmetic route.  This
    probe is reported as capability evidence only; it is never mixed into the
    speed result.
    """
    rows = min(len(y), 256)
    batch_size = min(rows, 32)
    core = NativeCore(core_path)
    model = NativeModel(
        task="binary_classification",
        features=x.shape[1],
        threads=1,
        seed=7,
        learning_rate=learning_rate,
        core=core,
    )
    try:
        loss = model.train_random_step(x[:rows], y[:rows], batch_size)
        trace = model.execution_trace()
    finally:
        model.close()
    required = {"c-abi", "cpp-dispatch", "rust-policy", "fortran-numeric", "zig-memory"}
    engines = set(trace["engines"])
    return {
        "purpose": "capability verification only; excluded from timing comparison",
        "rows": rows,
        "batch_size": batch_size,
        "loss": float(loss),
        "execution_trace": trace,
        "required_engines": sorted(required),
        "all_required_engines_observed": required.issubset(engines),
    }


def timed_native_preprocess(
    core_path: Path,
    source: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> dict[str, float]:
    """Time the public fused native preprocessing API, including its output copy."""
    core = NativeCore(core_path)
    started = time.perf_counter_ns()
    output, repaired = core.preprocess_f32(source, means, stds, clip_limit=4.0)
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
    return {
        "seconds": elapsed,
        "repaired_values": float(repaired),
        "checksum": float(np.sum(output, dtype=np.float64)),
    }


def timed_numpy_preprocess(
    source: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> dict[str, float]:
    """Reference the same impute → normalize → clip float32 semantics in NumPy."""
    started = time.perf_counter_ns()
    invalid = ~np.isfinite(source)
    output = np.where(invalid, means.reshape(1, -1), source).astype(np.float32, copy=False)
    output = (output - means) / stds
    np.clip(output, -4.0, 4.0, out=output)
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
    return {
        "seconds": elapsed,
        "repaired_values": float(np.count_nonzero(invalid)),
        "checksum": float(np.sum(output, dtype=np.float64)),
    }


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


def summarize_preprocess(samples: list[dict[str, float]], values: int) -> dict[str, Any]:
    seconds = [sample["seconds"] for sample in samples]
    median_seconds = statistics.median(seconds)
    return {
        "median_seconds": median_seconds,
        "values_per_second": values / median_seconds,
        "repaired_values": int(samples[-1]["repaired_values"]),
        "checksum": samples[-1]["checksum"],
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
    preprocess_source = x.copy()
    preprocess_source[::257, 0] = np.nan
    preprocess_source[::193, -1] = np.inf
    preprocess_means = np.ascontiguousarray(np.mean(x, axis=0, dtype=np.float64), dtype=np.float32)
    preprocess_stds = np.ascontiguousarray(np.std(x, axis=0, dtype=np.float64), dtype=np.float32)
    preprocess_stds = np.maximum(preprocess_stds, np.float32(1.0e-6))
    core = NativeCore(core_path)
    pipeline = native_pipeline_probe(
        core_path,
        x,
        y,
        learning_rate=args.learning_rate,
    )

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
    preprocess_runners: dict[str, Callable[[], dict[str, float]]] = {
        "kernelyra_native": lambda: timed_native_preprocess(
            core_path, preprocess_source, preprocess_means, preprocess_stds
        ),
        "numpy": lambda: timed_numpy_preprocess(
            preprocess_source, preprocess_means, preprocess_stds
        ),
    }
    for _ in range(args.warmups):
        for runner in runners.values():
            runner()
        for runner in preprocess_runners.values():
            runner()
    samples = {name: [] for name in runners}
    names = list(runners)
    for repeat in range(args.repeats):
        order = names[repeat % len(names):] + names[: repeat % len(names)]
        for name in order:
            samples[name].append(runners[name]())

    preprocess_samples = {name: [] for name in preprocess_runners}
    preprocess_names = list(preprocess_runners)
    for repeat in range(args.repeats):
        order = preprocess_names[repeat % len(preprocess_names):] + preprocess_names[: repeat % len(preprocess_names)]
        for name in order:
            preprocess_samples[name].append(preprocess_runners[name]())

    results = {name: summarize(values, args.steps) for name, values in samples.items()}
    preprocess_results = {
        name: summarize_preprocess(values, preprocess_source.size)
        for name, values in preprocess_samples.items()
    }
    native_seconds = results["kernelyra_native"]["median_seconds"]
    numpy_seconds = results["numpy"]["median_seconds"]
    preprocess_native_seconds = preprocess_results["kernelyra_native"]["median_seconds"]
    preprocess_numpy_seconds = preprocess_results["numpy"]["median_seconds"]
    payload = {
        "schema": "kernelyra-cpu-benchmark/2",
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
        "native_pipeline_probe": pipeline,
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
        "preprocess_workload": {
            "operation": "copy, impute non-finite values, normalize, clip to [-4, 4]",
            "rows": args.rows,
            "features": args.features,
            "dtype": "float32",
            "injected_nonfinite_values": int(np.count_nonzero(~np.isfinite(preprocess_source))),
            "repeats": args.repeats,
            "warmups": args.warmups,
            "measurement_order": "alternating",
        },
        "preprocess_results": preprocess_results,
        "preprocess_comparison": {
            "native_speedup_vs_numpy": preprocess_numpy_seconds / preprocess_native_seconds,
            "native_time_over_numpy": preprocess_native_seconds / preprocess_numpy_seconds,
            "winner_for_this_case": "kernelyra_native"
            if preprocess_native_seconds < preprocess_numpy_seconds
            else "numpy",
            "absolute_checksum_delta": abs(
                preprocess_results["kernelyra_native"]["checksum"]
                - preprocess_results["numpy"]["checksum"]
            ),
            "repaired_values_match": (
                preprocess_results["kernelyra_native"]["repaired_values"]
                == preprocess_results["numpy"]["repaired_values"]
            ),
        },
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
