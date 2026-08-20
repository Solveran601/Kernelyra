from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ..metrics import binary_metrics, multiclass_metrics, regression_metrics
from ..models import TaskType
from ..streaming import StreamingTabularSource
from .base import BackendConfig, EvaluationResult, StepResult, TrainingSession


class NumpyBackend:
    name = "numpy"
    version = "1.0"
    task_types = tuple(item.value for item in TaskType)
    metrics = (
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "macro_f1",
        "micro_f1",
        "mae",
        "rmse",
        "r2",
    )
    export_formats: tuple[str, ...] = ("npz", "run-manifest-json")

    def inspect_model(self, path: Path) -> dict[str, Any]:
        return {"backend": self.name, "valid": path.suffix.lower() == ".npz" and path.is_file()}

    def create_session(self, config: BackendConfig) -> TrainingSession:
        if config.task_type not in self.task_types:
            raise ValueError(f"NumPy backend не поддерживает task '{config.task_type}'")
        if config.precision not in {"auto", "float32", "float64"}:
            raise ValueError("NumPy backend supports precision=auto, float32 or float64")
        dtype = np.float64 if config.precision == "float64" else np.float32
        source = (
            StreamingTabularSource(config.dataset_spec, config.seed, config.data_workers, config.prefetch)
            if config.dataset_spec
            else None
        )
        rng = np.random.default_rng(config.seed)
        if source is not None:
            train_x = np.empty((0, int(source.spec["features"])), dtype=dtype)
            train_y = np.empty(0, dtype=dtype)
            validation_x = source.validation_x.astype(dtype)
            validation_y = source.validation_y.astype(dtype)
            test_x = source.test_x.astype(dtype)
            test_y = source.test_y.astype(dtype)
            all_y = np.concatenate((validation_y, test_y))
        else:
            if config.x is None or config.y is None:
                raise ValueError("NumPy backend requires arrays or a streaming dataset spec")
            x, y = config.x.astype(dtype), config.y.astype(dtype)
            order = rng.permutation(len(x))
            x, y = x[order], y[order]
            validation_size = max(16, int(round(len(x) * config.validation_fraction)))
            test_size = max(1, int(round(len(x) * config.test_fraction)))
            train_end = len(x) - validation_size - test_size
            if train_end < 8:
                raise ValueError("Недостаточно строк после deterministic train/validation/test split")
            train_x, validation_x = x[:train_end], x[train_end : train_end + validation_size]
            train_y, validation_y = y[:train_end], y[train_end : train_end + validation_size]
            test_x, test_y = x[train_end + validation_size :], y[train_end + validation_size :]
            all_y = y
        test_size = len(test_y)
        learning_rate = config.learning_rate or {
            "eco": .025,
            "low-memory": .025,
            "balanced": .035,
            "performance": .04,
            "workstation": .04,
        }.get(config.profile, .035)
        state: dict[str, Any] = {
            "task_type": config.task_type,
            "learning_rate": learning_rate,
            "weight_decay": float(config.weight_decay),
            "dtype": dtype,
        }
        if config.task_type == TaskType.MULTICLASS_CLASSIFICATION.value:
            class_count = int(max(all_y)) + 1
            state.update(
                {
                    "weights": rng.normal(0, .1, (validation_x.shape[1], class_count)),
                    "bias": np.zeros(class_count, dtype=dtype),
                    "class_count": class_count,
                }
            )
        else:
            state.update({"weights": rng.normal(0, .1, validation_x.shape[1]), "bias": 0.0})
        if config.task_type == TaskType.REGRESSION.value:
            target_mean = float(all_y.mean())
            target_std = float(all_y.std()) or 1.0
            state.update({"target_mean": target_mean, "target_std": target_std})
        session = TrainingSession(
            state,
            train_x,
            train_y,
            validation_x,
            validation_y,
            test_x,
            test_y,
            rng,
            metadata={
                "task_type": config.task_type,
                "split_seed": config.seed,
                "test_rows": test_size,
                "backend_version": self.version,
                "train_records": source.train_records if source else len(train_x),
                "streaming": source is not None,
                "precision": "float64" if dtype == np.float64 else "float32",
            },
            data_source=source,
        )
        if config.checkpoint_path and config.checkpoint_path.exists():
            self.restore_checkpoint(session, config.checkpoint_path)
        elif config.model_path:
            if not config.model_path.is_file():
                raise ValueError("NumPy model file was not found")
            self.restore_checkpoint(session, config.model_path)
        return session

    def train_step(self, session: TrainingSession, batch_size: int) -> StepResult:
        if session.data_source is not None:
            xb, yb = session.data_source.next_batch(batch_size)
            dtype = session.state["dtype"]
            xb, yb = xb.astype(dtype), yb.astype(dtype)
        else:
            idx = session.rng.integers(0, len(session.train_x), size=batch_size)
            xb, yb = session.train_x[idx], session.train_y[idx]
        weights, bias = session.state["weights"], session.state["bias"]
        lr = float(session.state["learning_rate"])
        decay = float(session.state["weight_decay"])
        task = str(session.state["task_type"])
        if task == TaskType.BINARY_CLASSIFICATION.value:
            logits = xb @ weights + bias
            probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
            error = probabilities - yb
            session.state["weights"] = weights - lr * (xb.T @ error / batch_size + decay * weights)
            session.state["bias"] = float(bias) - lr * float(error.mean())
            loss = float(
                -np.mean(yb * np.log(probabilities + 1e-8) + (1 - yb) * np.log(1 - probabilities + 1e-8))
            )
        elif task == TaskType.MULTICLASS_CLASSIFICATION.value:
            logits = xb @ weights + bias
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probabilities = exp / exp.sum(axis=1, keepdims=True)
            truth = yb.astype(np.int64)
            loss = float(-np.log(probabilities[np.arange(batch_size), truth] + 1e-8).mean())
            gradient = probabilities
            gradient[np.arange(batch_size), truth] -= 1
            gradient /= batch_size
            session.state["weights"] = weights - lr * (xb.T @ gradient + decay * weights)
            session.state["bias"] = bias - lr * gradient.sum(axis=0)
        else:
            normalized = (yb - session.state["target_mean"]) / session.state["target_std"]
            predicted = xb @ weights + bias
            error = predicted - normalized
            session.state["weights"] = weights - lr * (2 * xb.T @ error / batch_size + decay * weights)
            session.state["bias"] = float(bias) - lr * 2 * float(error.mean())
            loss = float(np.mean(error**2))
        if not np.isfinite(loss):
            raise FloatingPointError("NaN guard: loss перестал быть конечным")
        return StepResult(loss=loss, samples=batch_size)

    def evaluate(self, session: TrainingSession) -> EvaluationResult:
        return self._evaluate_arrays(session, session.validation_x, session.validation_y)

    def evaluate_test(self, session: TrainingSession) -> EvaluationResult:
        return self._evaluate_arrays(session, session.test_x, session.test_y)

    @staticmethod
    def close_session(session: TrainingSession) -> None:
        if session.data_source is not None:
            session.data_source.close()

    @staticmethod
    def _evaluate_arrays(session: TrainingSession, x: np.ndarray, y: np.ndarray) -> EvaluationResult:
        task = str(session.state["task_type"])
        if task == TaskType.BINARY_CLASSIFICATION.value:
            logits = x @ session.state["weights"] + session.state["bias"]
            probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
            loss = float(
                -np.mean(y * np.log(probabilities + 1e-8) + (1 - y) * np.log(1 - probabilities + 1e-8))
            )
            metrics = binary_metrics(y, probabilities, loss)
            return EvaluationResult(score=float(metrics["accuracy"]), metrics=metrics)
        if task == TaskType.MULTICLASS_CLASSIFICATION.value:
            logits = x @ session.state["weights"] + session.state["bias"]
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probabilities = exp / exp.sum(axis=1, keepdims=True)
            truth = y.astype(np.int64)
            loss = float(-np.log(probabilities[np.arange(len(y)), truth] + 1e-8).mean())
            metrics = multiclass_metrics(y, probabilities, loss)
            return EvaluationResult(score=float(metrics["accuracy"]), metrics=metrics)
        normalized = x @ session.state["weights"] + session.state["bias"]
        predicted = normalized * session.state["target_std"] + session.state["target_mean"]
        metrics = regression_metrics(y, predicted)
        return EvaluationResult(score=float(metrics["r2"]), metrics=metrics)

    def save_checkpoint(self, session: TrainingSession, path: Path, metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(path.stem + ".pending.npz")
        payload = {
            "weights": session.state["weights"],
            "bias": session.state["bias"],
            "task_type": session.state["task_type"],
            **metadata,
        }
        if session.data_source is not None:
            payload.update(session.data_source.state())
        if session.state["task_type"] == TaskType.REGRESSION.value:
            payload.update(
                {
                    "target_mean": session.state["target_mean"],
                    "target_std": session.state["target_std"],
                }
            )
        np.savez(pending, **payload)
        os.replace(pending, path)

    def restore_checkpoint(self, session: TrainingSession, path: Path) -> None:
        with np.load(path, allow_pickle=False) as saved:
            checkpoint_task = str(saved["task_type"]) if "task_type" in saved else "binary_classification"
            if checkpoint_task != session.state["task_type"]:
                raise ValueError("Checkpoint task_type несовместим с текущим run")
            weights = saved["weights"]
            if weights.shape != session.state["weights"].shape:
                raise ValueError("Checkpoint feature/class shape несовместим с текущим run")
            session.state["weights"] = weights.astype(float).copy()
            bias = saved["bias"]
            session.state["bias"] = float(bias) if bias.ndim == 0 else bias.astype(float).copy()
            if checkpoint_task == TaskType.REGRESSION.value:
                session.state["target_mean"] = float(saved["target_mean"])
                session.state["target_std"] = float(saved["target_std"])
            if session.data_source is not None and "stream_rows_consumed" in saved:
                session.data_source.restore_rows(int(saved["stream_rows_consumed"]))
