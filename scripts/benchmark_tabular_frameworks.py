"""Fairness-labelled CPU benchmark for Kernelyra and ten optional ML packages.

Linear-gradient runners share the same synthetic binary classification task.
Tree and online learners are measured separately because their algorithms do not
perform equivalent update work; their values are never used for a speed claim.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


@dataclass(frozen=True)
class Measurement:
    name: str
    group: str
    status: str
    seconds: float | None
    updates_per_second: float | None
    accuracy: float | None
    distribution_version: str | None
    note: str | None = None


def data(rows: int, features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, features)).astype(np.float32)
    weights = rng.normal(size=features).astype(np.float32)
    labels = (x @ weights + rng.normal(scale=.3, size=rows) > 0).astype(np.int64)
    return x, labels


def accuracy(predictions: Any, target: np.ndarray) -> float:
    return float((np.asarray(predictions).reshape(-1).astype(np.int64) == target).mean())


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def numpy_linear(x: np.ndarray, y: np.ndarray, steps: int, learning_rate: float, _: int) -> float:
    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = np.float32(0.0)
    target = y.astype(np.float32)
    for _ in range(steps):
        probabilities = sigmoid(x @ weights + bias)
        errors = probabilities - target
        weights -= learning_rate * (x.T @ errors / len(x))
        bias -= learning_rate * errors.mean(dtype=np.float32)
    return accuracy(sigmoid(x @ weights + bias) >= .5, y)


def kernelyra_native(x: np.ndarray, y: np.ndarray, steps: int, learning_rate: float, seed: int) -> float:
    from kernelyra.native_core import NativeModel

    with NativeModel(
        task="binary_classification", features=x.shape[1], learning_rate=learning_rate, seed=seed, threads=1
    ) as model:
        # Match NumPy's deterministic full-batch update, not the fast random
        # batch API used by the production streaming scheduler.
        model.import_parameters(np.zeros(x.shape[1], dtype=np.float32), np.asarray([0.0], dtype=np.float32))
        for _ in range(steps):
            model.train_step(x, y.astype(np.float32))
        return accuracy(model.predict(x) >= .5, y)


def torch_linear(x: np.ndarray, y: np.ndarray, steps: int, learning_rate: float, seed: int) -> float:
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    inputs = torch.from_numpy(x)
    labels = torch.from_numpy(y.astype(np.float32))
    model = torch.nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(inputs).reshape(-1), labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return accuracy(torch.sigmoid(model(inputs).reshape(-1)).numpy() >= .5, y)


def tensorflow_linear(x: np.ndarray, y: np.ndarray, steps: int, learning_rate: float, seed: int) -> float:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
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

    update()
    for _ in range(steps - 1):
        update()
    return accuracy(tf.math.sigmoid(tf.linalg.matvec(inputs, weights) + bias).numpy() >= .5, y)


def jax_linear(x: np.ndarray, y: np.ndarray, steps: int, learning_rate: float, _: int) -> float:
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
    predictions = jax.nn.sigmoid(inputs @ weights + bias).block_until_ready()
    return accuracy(np.asarray(predictions) >= .5, y)


def flax_optax_linear(x: np.ndarray, y: np.ndarray, steps: int, learning_rate: float, seed: int) -> float:
    import flax.linen as nn
    from flax.training import train_state
    import jax
    import jax.numpy as jnp
    import optax

    class Linear(nn.Module):
        @nn.compact
        def __call__(self, values: Any) -> Any:
            return nn.Dense(1)(values).reshape(-1)

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
            return optax.sigmoid_binary_cross_entropy(current.apply_fn(params, inputs), labels).mean()

        gradients = jax.grad(loss)(current.params)
        return current.apply_gradients(grads=gradients)

    for _ in range(steps):
        state = update(state)
    predictions = jax.nn.sigmoid(state.apply_fn(state.params, inputs)).block_until_ready()
    return accuracy(np.asarray(predictions) >= .5, y)


def sklearn_linear(x: np.ndarray, y: np.ndarray, steps: int, learning_rate: float, seed: int) -> float:
    from sklearn.linear_model import SGDClassifier

    model = SGDClassifier(
        loss="log_loss", alpha=0.0, learning_rate="constant", eta0=learning_rate, random_state=seed, tol=None
    )
    for index in range(steps):
        model.partial_fit(x, y, classes=np.asarray([0, 1]) if index == 0 else None)
    return accuracy(model.predict(x), y)


def river_online(x: np.ndarray, y: np.ndarray, steps: int, learning_rate: float, _: int) -> float:
    from river import linear_model, optim

    model = linear_model.LogisticRegression(optimizer=optim.SGD(learning_rate))
    for _ in range(steps):
        for values, label in zip(x, y, strict=True):
            model.learn_one({str(index): float(value) for index, value in enumerate(values)}, int(label))
    predictions = [model.predict_one({str(index): float(value) for index, value in enumerate(values)}) or False for values in x]
    return accuracy(predictions, y)


def xgboost_trees(x: np.ndarray, y: np.ndarray, steps: int, _: float, seed: int) -> float:
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=steps, max_depth=4, learning_rate=.1, n_jobs=1, random_state=seed,
        tree_method="hist", eval_metric="logloss",
    )
    model.fit(x, y)
    return accuracy(model.predict(x), y)


def lightgbm_trees(x: np.ndarray, y: np.ndarray, steps: int, _: float, seed: int) -> float:
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=steps, max_depth=4, learning_rate=.1, n_jobs=1, random_state=seed, verbosity=-1
    )
    model.fit(x, y)
    return accuracy(model.predict(x), y)


def catboost_trees(x: np.ndarray, y: np.ndarray, steps: int, _: float, seed: int) -> float:
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(iterations=steps, depth=4, learning_rate=.1, thread_count=1, random_seed=seed, verbose=False)
    model.fit(x, y)
    return accuracy(model.predict(x), y)


RUNNERS: dict[str, tuple[str, str | None, Callable[[np.ndarray, np.ndarray, int, float, int], float]]] = {
    "kernelyra_native": ("matched_linear", "kernelyra-ai", kernelyra_native),
    "numpy": ("matched_linear", "numpy", numpy_linear),
    "torch": ("matched_linear", "torch", torch_linear),
    "tensorflow": ("matched_linear", "tensorflow", tensorflow_linear),
    "jax": ("matched_linear", "jax", jax_linear),
    "flax_optax": ("matched_linear", "flax", flax_optax_linear),
    "scikit_learn": ("matched_linear", "scikit-learn", sklearn_linear),
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


def benchmark(name: str, group: str, distribution: str | None, runner: Callable[..., float], args: argparse.Namespace, x: np.ndarray, y: np.ndarray) -> Measurement:
    try:
        values: list[tuple[float, float]] = []
        for _ in range(args.runs):
            started = time.perf_counter()
            score = runner(x, y, args.steps, args.learning_rate, args.seed)
            values.append((time.perf_counter() - started, score))
        seconds = statistics.median(item[0] for item in values)
        return Measurement(name, group, "ok", seconds, args.steps / seconds, statistics.median(item[1] for item in values), version(distribution))
    except ModuleNotFoundError as error:
        return Measurement(name, group, "missing", None, None, None, version(distribution), str(error))
    except Exception as error:  # A framework failure must remain visible in the report.
        return Measurement(name, group, "error", None, None, None, version(distribution), f"{type(error).__name__}: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Kernelyra framework benchmark matrix")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=.03)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--only", nargs="*", choices=sorted(RUNNERS))
    parser.add_argument("--output", type=Path, default=Path(".benchmarks/framework-matrix.json"))
    args = parser.parse_args()
    if min(args.rows, args.features, args.steps, args.runs) < 1 or args.learning_rate <= 0:
        raise SystemExit("rows, features, steps, runs and learning-rate must be positive")
    x, y = data(args.rows, args.features, args.seed)
    names = args.only or list(RUNNERS)
    results = [benchmark(name, *RUNNERS[name], args, x, y) for name in names]
    payload = {
        "contract": "kernelyra-framework-benchmark/1",
        "workload": {key: getattr(args, key) for key in ("rows", "features", "steps", "runs", "learning_rate", "seed")},
        "results": [asdict(item) for item in results],
        "comparison_rule": "Only matched_linear rows execute the same full-batch logistic-regression update. Tree and online rows are capability measurements, not speed comparisons.",
        "installed_training_dependencies": ["torch", "tensorflow", "jax", "flax", "optax", "scikit-learn", "xgboost", "lightgbm", "catboost", "river"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(item.status in {"ok", "missing"} for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
