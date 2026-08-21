"""Measure the native binary-training kernel against an equivalent NumPy baseline.

This benchmark makes a deliberately narrow claim: it compares the same full-batch
logistic-regression update on the same synthetic, linearly separable float32 data.
It reports a pass only if native median throughput is at least 2x while native
accuracy is not lower by more than one percentage point.  It does not claim a
universal TensorFlow/PyTorch/LLM victory.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

from kernelyra.native_core import NativeModel, native_core_status


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def working_set_bytes() -> int | None:
    if os.name != "nt":
        return None
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    get_process_memory_info.restype = ctypes.c_int
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.WorkingSetSize)


def _next_random(state: int) -> int:
    state ^= state >> 12
    state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 27
    return (state * 2685821657736338717) & 0xFFFFFFFFFFFFFFFF


def native_initial_weights(features: int, seed: int) -> np.ndarray:
    state = seed or 0x9E3779B97F4A7C15
    weights = np.empty(features, dtype=np.float32)
    for index in range(features):
        state = _next_random(state)
        weights[index] = (((state >> 40) & 0xFFFFFF) / 16777216.0 - 0.5) * 0.2
    return weights


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def make_data(rows: int, features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (rows, features)).astype(np.float32)
    teacher = rng.normal(0.0, 1.0, features).astype(np.float32)
    logits = x @ teacher + rng.normal(0.0, 0.35, rows).astype(np.float32)
    return x, (logits > 0.0).astype(np.float32)


@dataclass(frozen=True)
class Measurement:
    name: str
    seconds: float
    updates_per_second: float
    accuracy: float
    peak_extra_working_set_bytes: int | None


def run_numpy(x: np.ndarray, y: np.ndarray, *, steps: int, lr: float, seed: int) -> Measurement:
    baseline_memory = working_set_bytes()
    weights = native_initial_weights(x.shape[1], seed)
    bias = np.float32(0.0)
    peak_memory = working_set_bytes()
    started = time.perf_counter()
    for _ in range(steps):
        probabilities = sigmoid(x @ weights + bias)
        errors = probabilities - y
        weights -= np.float32(lr) * (x.T @ errors / np.float32(len(y)))
        bias -= np.float32(lr) * errors.mean(dtype=np.float32)
        peak_memory = max(peak_memory or 0, working_set_bytes() or 0) or None
    seconds = time.perf_counter() - started
    accuracy = float(((sigmoid(x @ weights + bias) >= 0.5) == y).mean())
    delta = None if baseline_memory is None or peak_memory is None else max(0, peak_memory - baseline_memory)
    return Measurement("numpy_vectorized", seconds, steps / seconds, accuracy, delta)


def run_native(x: np.ndarray, y: np.ndarray, *, steps: int, lr: float, seed: int, threads: int) -> Measurement:
    baseline_memory = working_set_bytes()
    model = NativeModel(
        task="binary_classification", features=x.shape[1], seed=seed,
        learning_rate=lr, weight_decay=0.0, threads=threads,
    )
    try:
        # The baseline uses this exact initialization.  Import it explicitly
        # instead of assuming the independent native RNG stays bit-identical.
        model.import_parameters(native_initial_weights(x.shape[1], seed), np.float32(0.0))
        peak_memory = working_set_bytes()
        started = time.perf_counter()
        for _ in range(steps):
            model.train_step(x, y)
            peak_memory = max(peak_memory or 0, working_set_bytes() or 0) or None
        seconds = time.perf_counter() - started
        accuracy = float(((model.predict(x) >= 0.5) == y).mean())
        delta = None if baseline_memory is None or peak_memory is None else max(0, peak_memory - baseline_memory)
        return Measurement("kernelyra_native", seconds, steps / seconds, accuracy, delta)
    finally:
        model.close()


def median_measurement(runs: list[Measurement]) -> Measurement:
    middle = sorted(runs, key=lambda item: item.seconds)[len(runs) // 2]
    return Measurement(
        middle.name,
        statistics.median(item.seconds for item in runs),
        statistics.median(item.updates_per_second for item in runs),
        statistics.median(item.accuracy for item in runs),
        statistics.median(item.peak_extra_working_set_bytes or 0 for item in runs) or None,
    )


def invoke_worker(args: argparse.Namespace, name: str) -> Measurement:
    command = [
        sys.executable, str(__file__), "--worker", name, "--rows", str(args.rows),
        "--features", str(args.features), "--steps", str(args.steps), "--threads", str(args.threads),
        "--learning-rate", str(args.learning_rate), "--seed", str(args.seed),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return Measurement(**json.loads(completed.stdout))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--features", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--worker", choices=("numpy", "native"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    x, y = make_data(args.rows, args.features, args.seed)
    if args.worker:
        if args.worker == "numpy":
            run_numpy(x, y, steps=2, lr=args.learning_rate, seed=args.seed)
            result = run_numpy(x, y, steps=args.steps, lr=args.learning_rate, seed=args.seed)
        else:
            run_native(x, y, steps=2, lr=args.learning_rate, seed=args.seed, threads=args.threads)
            result = run_native(x, y, steps=args.steps, lr=args.learning_rate, seed=args.seed, threads=args.threads)
        print(json.dumps(asdict(result), sort_keys=True))
        return 0
    # One warm-up outside measured runs avoids counting DLL dispatch/allocation costs.
    numpy_result = median_measurement([
        invoke_worker(args, "numpy") for _ in range(args.runs)
    ])
    native_result = median_measurement([
        invoke_worker(args, "native") for _ in range(args.runs)
    ])
    speedup = native_result.updates_per_second / numpy_result.updates_per_second
    accuracy_gap = native_result.accuracy - numpy_result.accuracy
    payload = {
        "workload": {"rows": args.rows, "features": args.features, "steps": args.steps, "runs": args.runs},
        "native_core": native_core_status(),
        "results": [asdict(numpy_result), asdict(native_result)],
        "speedup_native_over_numpy": speedup,
        "accuracy_gap_native_minus_numpy": accuracy_gap,
        "pass": speedup >= 2.0 and accuracy_gap >= -0.01 and (
            native_result.peak_extra_working_set_bytes is None or numpy_result.peak_extra_working_set_bytes is None or
            native_result.peak_extra_working_set_bytes < numpy_result.peak_extra_working_set_bytes
        ),
        "criterion": "native >= 2x NumPy updates/s, accuracy no worse than 1 percentage point, and smaller extra working set",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
