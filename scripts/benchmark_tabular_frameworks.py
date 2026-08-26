"""Reproducible CPU benchmark matrix for Kernelyra and ten ML libraries.

The script deliberately separates identical full-batch logistic-regression
implementations from online and tree learners. Only the
``full_batch_logistic_regression`` group may be compared for speed. Every
runner is trained on one synthetic split and evaluated on a distinct hold-out
split, so reported accuracy is not training accuracy.

Set ``KERNELYRA_BENCH_THREADS`` before starting Python to select a shared CPU
thread cap. One thread is the default because it is the most portable and
least affected by different BLAS/OpenMP defaults.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

THREAD_LIMIT = int(os.environ.get("KERNELYRA_BENCH_THREADS", "1"))
if THREAD_LIMIT < 1:
    raise SystemExit("KERNELYRA_BENCH_THREADS must be a positive integer")

# These must be set before importing NumPy or an optional framework.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = str(THREAD_LIMIT)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("XLA_FLAGS", f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={THREAD_LIMIT}")

import numpy as np  # noqa: E402 - thread variables must precede this import.


@dataclass(frozen=True)
class Measurement:
    name: str
    group: str
    status: str
    median_seconds: float | None
    updates_per_second: float | None
    holdout_accuracy: float | None
    distribution_version: str | None
    note: str | None = None


Runner = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, int, int], float]


def make_data(train_rows: int, evaluation_rows: int, features: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create distinct train/evaluation samples from the same hidden rule."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=features).astype(np.float32)
    bias = np.float32(rng.normal())

    def sample(rows: int) -> tuple[np.ndarray, np.ndarray]:
        values = rng.normal(size=(rows, features)).astype(np.float32)
        labels = (values @ weights + bias + rng.normal(scale=.3, size=rows) > 0).astype(np.int64)
        return values, labels

    train_x, train_y = sample(train_rows)
    evaluation_x, evaluation_y = sample(evaluation_rows)
    return train_x, train_y, evaluation_x, evaluation_y


def accuracy(predictions: Any, target: np.ndarray) -> float:
    return float((np.asarray(predictions).reshape(-1).astype(np.int64) == target).mean())


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def numpy_linear(x: np.ndarray, y: np.ndarray, evaluation_x: np.ndarray, evaluation_y: np.ndarray, steps: int, learning_rate: float, _: int, __: int) -> float:
    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = np.float32(0.0)
    target = y.astype(np.float32)
    for _ in range(steps):
        errors = sigmoid(x @ weights + bias) - target
        weights -= learning_rate * (x.T @ errors / len(x))
        bias -= learning_rate * errors.mean(dtype=np.float32)
    return accuracy(sigmoid(evaluation_x @ weights + bias) >= .5, evaluation_y)


def kernelyra_native(x: np.ndarray, y: np.ndarray, evaluation_x: np.ndarray, evaluation_y: np.ndarray, steps: int, learning_rate: float, seed: int, threads: int) -> float:
    from kernelyra.native_core import NativeModel

    with NativeModel(task="binary_classification", features=x.shape[1], learning_rate=learning_rate, seed=seed, threads=threads) as model:
        model.import_parameters(np.zeros(x.shape[1], dtype=np.float32), np.asarray([0.0], dtype=np.float32))
        for _ in range(steps):
            model.train_step(x, y.astype(np.float32))
        return accuracy(model.predict(evaluation_x) >= .5, evaluation_y)


def torch_linear(x: np.ndarray, y: np.ndarray, evaluation_x: np.ndarray, evaluation_y: np.ndarray, steps: int, learning_rate: float, seed: int, threads: int) -> float:
    import torch

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    inputs = torch.from_numpy(x)
    labels = torch.from_numpy(y.astype(np.float32))
    model = torch.nn.Linear(x.shape[1], 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(inputs).reshape(-1), labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predictions = torch.sigmoid(model(torch.from_numpy(evaluation_x)).reshape(-1)).numpy() >= .5
    return accuracy(predictions, evaluation_y)


def tensorflow_linear(x: np.ndarray, y: np.ndarray, evaluation_x: np.ndarray, evaluation_y: np.ndarray, steps: int, learning_rate: float, seed: int, threads: int) -> float:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(threads)
    except RuntimeError:
        pass  # TensorFlow permits this only before its runtime is initialised.
    inputs = tf.convert_to_tensor(x)
    labels = tf.convert_to_tensor(y.astype(np.float32))
    weights = tf.Variable(tf.zeros((x.shape[1],), dtype=tf.float32))
    bias = tf.Variable(0.0, dtype=tf.float32)
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    @tf.function(reduce_retracing=True)
    def update() -> None:
        with tf.GradientTape() as tape:
            logits = tf.linalg.matvec(inputs, weights) + bias
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        optimizer.apply_gradients(zip(tape.gradient(loss, (weights, bias)), (weights, bias), strict=True))

    for _ in range(steps):
        update()
    return accuracy(tf.math.sigmoid(tf.linalg.matvec(evaluation_x, weights) + bias).numpy() >= .5, evaluation_y)


def jax_linear(x: np.ndarray, y: np.ndarray, evaluation_x: np.ndarray, evaluation_y: np.ndarray, steps: int, learning_rate: float, _: int, __: int) -> float:
    import jax
    import jax.numpy as jnp

    inputs, labels = jnp.asarray(x), jnp.asarray(y, dtype=jnp.float32)

    @jax.jit
    def update(weights: Any, bias: Any) -> tuple[Any, Any]:
        errors = jax.nn.sigmoid(inputs @ weights + bias) - labels
        return weights - learning_rate * (inputs.T @ errors / inputs.shape[0]), bias - learning_rate * jnp.mean(errors)

    weights, bias = jnp.zeros(x.shape[1], dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)
    for _ in range(steps):
        weights, bias = update(weights, bias)
    predictions = jax.nn.sigmoid(jnp.asarray(evaluation_x) @ weights + bias).block_until_ready()
    return accuracy(np.asarray(predictions) >= .5, evaluation_y)


def flax_optax_linear(x: np.ndarray, y: np.ndarray, evaluation_x: np.ndarray, evaluation_y: np.ndarray, steps: int, learning_rate: float, seed: int, _: int) -> float:
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state

    class Linear(nn.Module):
        @nn.compact
        def __call__(self, values: Any) -> Any:
            return nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(values).reshape(-1)

    inputs, labels = jnp.asarray(x), jnp.asarray(y, dtype=jnp.float32)
    model = Linear()
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.key(seed), inputs)["params"],
        tx=optax.sgd(learning_rate),
    )

    @jax.jit
    def update(current: Any) -> Any:
        def loss(params: Any) -> Any:
            return optax.sigmoid_binary_cross_entropy(current.apply_fn({"params": params}, inputs), labels).mean()

        return current.apply_gradients(grads=jax.grad(loss)(current.params))

    for _ in range(steps):
        state = update(state)
    predictions = jax.nn.sigmoid(state.apply_fn({"params": state.params}, jnp.asarray(evaluation_x))).block_until_ready()
    return accuracy(np.asarray(predictions) >= .5, evaluation_y)


def sklearn_linear(x: np.ndarray, y: np.ndarray, evaluation_x: np.ndarray, evaluation_y: np.ndarray, steps: int, learning_rate: float, seed: int, _: int) -> float:
    from sklearn.linear_model import SGDClassifier

    model = SGDClassifier(loss="log_loss", alpha=1e-8, learning_rate="constant", eta0=learning_rate, random_state=seed, tol=None)
    for index in range(steps):
        model.partial_fit(x, y, classes=np.asarray([0, 1]) if index == 0 else None)
    return accuracy(model.predict(evaluation_x), evaluation_y)


def river_online(x: np.ndarray, y: np.ndarray, evaluation_x: np.ndarray, evaluation_y: np.ndarray, steps: int, learning_rate: float, _: int, __: int) -> float:
    from river import linear_model, optim

    model = linear_model.LogisticRegression(optimizer=optim.SGD(learning_rate))
    for _ in range(steps):
        for values, label in zip(x, y, strict=True):
            model.learn_one({str(index): float(value) for index, value in enumerate(values)}, int(label))
    predictions = [model.predict_one({str(index): float(value) for index, value in enumerate(values)}) or False for values in evaluation_x]
    return accuracy(predictions, evaluation_y)


def xgboost_trees(x: np.ndarray, y: np.ndarray, evaluation_x: np.ndarray, evaluation_y: np.ndarray, steps: int, _: float, seed: int, threads: int) -> float:
    from xgboost import XGBClassifier

    model = XGBClassifier(n_estimators=steps, max_depth=4, learning_rate=.1, n_jobs=threads, random_state=seed, tree_method="hist", eval_metric="logloss")
    model.fit(x, y)
    return accuracy(model.predict(evaluation_x), evaluation_y)


def lightgbm_trees(x: np.ndarray, y: np.ndarray, evaluation_x: np.ndarray, evaluation_y: np.ndarray, steps: int, _: float, seed: int, threads: int) -> float:
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(n_estimators=steps, max_depth=4, learning_rate=.1, n_jobs=threads, random_state=seed, verbosity=-1)
    model.fit(x, y)
    return accuracy(model.predict(evaluation_x), evaluation_y)


def catboost_trees(x: np.ndarray, y: np.ndarray, evaluation_x: np.ndarray, evaluation_y: np.ndarray, steps: int, _: float, seed: int, threads: int) -> float:
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(iterations=steps, depth=4, learning_rate=.1, thread_count=threads, random_seed=seed, verbose=False)
    model.fit(x, y)
    return accuracy(model.predict(evaluation_x), evaluation_y)


RUNNERS: dict[str, tuple[str, str | None, Runner]] = {
    "kernelyra_native": ("full_batch_logistic_regression", "kernelyra-ai", kernelyra_native),
    "numpy": ("full_batch_logistic_regression", "numpy", numpy_linear),
    "torch": ("full_batch_logistic_regression", "torch", torch_linear),
    "tensorflow": ("full_batch_logistic_regression", "tensorflow", tensorflow_linear),
    "jax": ("full_batch_logistic_regression", "jax", jax_linear),
    "flax_optax": ("full_batch_logistic_regression", "flax", flax_optax_linear),
    "scikit_learn": ("linear_task_not_identical", "scikit-learn", sklearn_linear),
    "river": ("online_linear_not_matched", "river", river_online),
    "xgboost": ("tree_not_matched", "xgboost", xgboost_trees),
    "lightgbm": ("tree_not_matched", "lightgbm", lightgbm_trees),
    "catboost": ("tree_not_matched", "catboost", catboost_trees),
}


def version(distribution: str | None) -> str | None:
    if distribution is None:
        return None
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def benchmark(name: str, group: str, distribution: str | None, runner: Runner, args: argparse.Namespace, train_x: np.ndarray, train_y: np.ndarray, evaluation_x: np.ndarray, evaluation_y: np.ndarray) -> Measurement:
    try:
        for _ in range(args.warmup_runs):
            runner(train_x, train_y, evaluation_x, evaluation_y, args.steps, args.learning_rate, args.seed, THREAD_LIMIT)
        values: list[tuple[float, float]] = []
        for _ in range(args.runs):
            started = time.perf_counter()
            score = runner(train_x, train_y, evaluation_x, evaluation_y, args.steps, args.learning_rate, args.seed, THREAD_LIMIT)
            values.append((time.perf_counter() - started, score))
        seconds = statistics.median(item[0] for item in values)
        return Measurement(name, group, "ok", seconds, args.steps / seconds, statistics.median(item[1] for item in values), version(distribution))
    except ModuleNotFoundError as error:
        return Measurement(name, group, "missing", None, None, None, version(distribution), str(error))
    except Exception as error:  # A framework failure must remain visible in the report.
        return Measurement(name, group, "error", None, None, None, version(distribution), f"{type(error).__name__}: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Kernelyra reproducible CPU framework benchmark matrix")
    parser.add_argument("--rows", type=int, default=8192, help="Training rows")
    parser.add_argument("--evaluation-rows", type=int, default=2048, help="Hold-out rows")
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--runs", type=int, default=3, help="Independent measured repetitions")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=.03)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--only", nargs="*", choices=sorted(RUNNERS))
    parser.add_argument("--output", type=Path, default=Path(".benchmarks/framework-matrix.json"))
    args = parser.parse_args()
    if min(args.rows, args.evaluation_rows, args.features, args.steps, args.runs) < 1 or args.warmup_runs < 0 or args.learning_rate <= 0:
        raise SystemExit("row counts, features, steps and runs must be positive; warmup-runs cannot be negative")

    train_x, train_y, evaluation_x, evaluation_y = make_data(args.rows, args.evaluation_rows, args.features, args.seed)
    names = args.only or list(RUNNERS)
    results = [benchmark(name, *RUNNERS[name], args, train_x, train_y, evaluation_x, evaluation_y) for name in names]
    payload = {
        "contract": "kernelyra-framework-benchmark/2",
        "executed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "os": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(), "thread_limit": THREAD_LIMIT, "gpu_disabled": True,
        },
        "workload": {
            "training_rows": args.rows, "holdout_rows": args.evaluation_rows, "features": args.features,
            "steps": args.steps, "warmup_runs": args.warmup_runs, "measured_runs": args.runs,
            "learning_rate": args.learning_rate, "seed": args.seed,
        },
        "results": [asdict(item) for item in results],
        "comparison_rule": (
            "Only full_batch_logistic_regression rows run the same full-batch float32 logistic-regression task and may be compared for speed. "
            "Every holdout_accuracy value is evaluated on a separate sample drawn from the same hidden rule. "
            "Tree and online rows are capability measurements and must not be placed in a speed ranking against the linear group."
        ),
        "timing_policy": (
            "Each reported time covers model construction, the requested training steps, and hold-out prediction. "
            "One unrecorded whole-run warm-up precedes the independent measurements; compilation that a framework performs "
            "inside a newly constructed measured run remains included rather than being hidden."
        ),
        "installed_training_dependencies": ["torch", "tensorflow", "jax", "flax", "optax", "scikit-learn", "xgboost", "lightgbm", "catboost", "river"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(item.status in {"ok", "missing"} for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
