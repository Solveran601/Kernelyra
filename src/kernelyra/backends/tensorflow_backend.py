from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..metrics import binary_metrics, multiclass_metrics, regression_metrics
from ..models import TaskType
from ..streaming import StreamingTabularSource
from .base import BackendConfig, EvaluationResult, StepResult, TrainingSession


def _tensorflow() -> Any:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    import tensorflow as tf

    return tf


class TensorFlowBackend:
    name = "tensorflow"
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
    export_formats = ("keras", "npz-checkpoint", "run-manifest-json")

    def inspect_model(self, path: Path) -> dict[str, Any]:
        valid = path.is_file() and path.suffix.lower() in {".keras", ".h5", ".hdf5"}
        return {"backend": self.name, "valid": valid, "fine_tune": valid}

    def create_session(self, config: BackendConfig) -> TrainingSession:
        tf = _tensorflow()
        gpu_limit = int(config.resource_limits.get("gpu_memory_mb") or 0)
        gpu_enabled = bool(config.resource_limits.get("gpu_enabled"))
        gpus = tf.config.list_physical_devices("GPU")
        if not gpu_enabled and gpus:
            tf.config.set_visible_devices([], "GPU")
            gpus = []
        for gpu in gpus:
            try:
                if gpu_limit > 0:
                    tf.config.set_logical_device_configuration(
                        gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_limit)]
                    )
                else:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as error:
                raise ValueError(f"TensorFlow GPU configuration недоступна: {error}") from None

        tf.keras.utils.set_random_seed(config.seed)
        requested_precision = config.precision
        if requested_precision not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("TensorFlow precision must be auto, float32, float16 or bfloat16")
        precision = "float32" if requested_precision == "auto" else requested_precision
        policy = {"float16": "mixed_float16", "bfloat16": "mixed_bfloat16"}.get(precision, "float32")
        tf.keras.mixed_precision.set_global_policy(policy)
        source = (
            StreamingTabularSource(config.dataset_spec, config.seed, config.data_workers, config.prefetch)
            if config.dataset_spec
            else None
        )
        rng = np.random.default_rng(config.seed)
        if source is not None:
            train_x = np.empty((0, int(source.spec["features"])), dtype=np.float32)
            train_y = np.empty(0, dtype=np.float32)
            validation_x, validation_y = source.validation_x, source.validation_y
            test_x, test_y = source.test_x, source.test_y
            all_y = np.concatenate((validation_y, test_y))
        else:
            if config.x is None or config.y is None:
                raise ValueError("TensorFlow backend requires arrays or a streaming dataset spec")
            x, y = config.x.astype(np.float32), config.y
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
        widths = list(config.hidden_layers) if config.hidden_layers else {
            "eco": [16, 8],
            "low-memory": [16, 8],
            "balanced": [32, 16],
            "performance": [64, 32, 16],
            "workstation": [128, 64, 32],
        }.get(config.profile, [32, 16])
        class_count = int(max(all_y)) + 1 if config.task_type == TaskType.MULTICLASS_CLASSIFICATION.value else 1

        if config.model_path:
            model = tf.keras.models.load_model(config.model_path, compile=False, safe_mode=True)
            input_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
            output_shape = model.output_shape[0] if isinstance(model.output_shape, list) else model.output_shape
            expected_output = class_count
            if not input_shape or input_shape[-1] != validation_x.shape[1]:
                raise ValueError("Keras model input shape несовместим с dataset schema")
            if not output_shape or output_shape[-1] != expected_output:
                raise ValueError("Keras model output shape несовместим с task type")
        else:
            layers: list[Any] = [tf.keras.layers.Input(shape=(validation_x.shape[1],))]
            layers.extend(tf.keras.layers.Dense(width, activation="relu") for width in widths)
            if config.task_type == TaskType.BINARY_CLASSIFICATION.value:
                layers.append(tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32"))
            elif config.task_type == TaskType.MULTICLASS_CLASSIFICATION.value:
                layers.append(tf.keras.layers.Dense(class_count, activation="softmax", dtype="float32"))
            else:
                layers.append(tf.keras.layers.Dense(1, dtype="float32"))
            model = tf.keras.Sequential(layers)
            model(np.zeros((1, validation_x.shape[1]), dtype=np.float32))

        if config.task_type == TaskType.BINARY_CLASSIFICATION.value:
            loss_fn = tf.keras.losses.BinaryCrossentropy()
        elif config.task_type == TaskType.MULTICLASS_CLASSIFICATION.value:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        else:
            loss_fn = tf.keras.losses.MeanSquaredError()
        learning_rate = config.learning_rate or {
            "eco": .0015,
            "low-memory": .0015,
            "balanced": .002,
            "performance": .0025,
            "workstation": .0025,
        }.get(config.profile, .002)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=config.weight_decay)
        if precision == "float16":
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        state = {
            "tf": tf,
            "model": model,
            "optimizer": optimizer,
            "loss_fn": loss_fn,
            "task_type": config.task_type,
            "class_count": class_count,
            "precision": precision,
        }
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
                "gpu_devices": len(gpus),
                "train_records": source.train_records if source else len(train_x),
                "streaming": source is not None,
                "precision": precision,
            },
            data_source=source,
        )
        if config.checkpoint_path and config.checkpoint_path.exists():
            self.restore_checkpoint(session, config.checkpoint_path)
        return session

    def train_step(self, session: TrainingSession, batch_size: int) -> StepResult:
        tf, model = session.state["tf"], session.state["model"]
        if session.data_source is not None:
            xb, yb = session.data_source.next_batch(batch_size)
        else:
            indices = session.rng.integers(0, len(session.train_x), size=batch_size)
            xb, yb = session.train_x[indices], session.train_y[indices]
        task = session.state["task_type"]
        with tf.GradientTape() as tape:
            output = model(xb, training=True)
            if task == TaskType.MULTICLASS_CLASSIFICATION.value:
                loss = session.state["loss_fn"](yb.astype(np.int64), output)
            else:
                loss = session.state["loss_fn"](yb.astype(np.float32)[:, None], output)
            optimizer = session.state["optimizer"]
            if hasattr(optimizer, "scale_loss"):
                gradient_loss = optimizer.scale_loss(loss)
            elif hasattr(optimizer, "get_scaled_loss"):
                gradient_loss = optimizer.get_scaled_loss(loss)
            else:
                gradient_loss = loss
        gradients = tape.gradient(gradient_loss, model.trainable_variables)
        if hasattr(optimizer, "get_unscaled_gradients"):
            gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables, strict=False))
        loss_value = float(loss.numpy())
        if not np.isfinite(loss_value):
            raise FloatingPointError("NaN guard: loss перестал быть конечным")
        return StepResult(loss=loss_value, samples=batch_size)

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
        output = session.state["model"](x, training=False).numpy()
        task = session.state["task_type"]
        if task == TaskType.BINARY_CLASSIFICATION.value:
            probabilities = output.reshape(-1)
            loss = float(
                -np.mean(y * np.log(probabilities + 1e-8) + (1 - y) * np.log(1 - probabilities + 1e-8))
            )
            metrics = binary_metrics(y, probabilities, loss)
            return EvaluationResult(float(metrics["accuracy"]), metrics)
        if task == TaskType.MULTICLASS_CLASSIFICATION.value:
            truth = y.astype(np.int64)
            loss = float(-np.log(output[np.arange(len(y)), truth] + 1e-8).mean())
            metrics = multiclass_metrics(y, output, loss)
            return EvaluationResult(float(metrics["accuracy"]), metrics)
        metrics = regression_metrics(y, output.reshape(-1))
        return EvaluationResult(float(metrics["r2"]), metrics)

    def save_checkpoint(self, session: TrainingSession, path: Path, metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(path.stem + ".pending.npz")
        values = {f"weight_{index}": value for index, value in enumerate(session.state["model"].get_weights())}
        stream_state = session.data_source.state() if session.data_source is not None else {}
        payload = {
            **metadata,
            **stream_state,
            "task_type": session.state["task_type"],
            **values,
        }
        np.savez(pending, **payload)
        for attempt in range(6):
            try:
                os.replace(pending, path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(.05 * (attempt + 1))

    def restore_checkpoint(self, session: TrainingSession, path: Path) -> None:
        model = session.state["model"]
        with np.load(path, allow_pickle=False) as saved:
            if "task_type" in saved and str(saved["task_type"]) != session.state["task_type"]:
                raise ValueError("Checkpoint task_type несовместим с текущим run")
            keys = [f"weight_{index}" for index in range(len(model.get_weights()))]
            if not all(key in saved for key in keys):
                raise ValueError("Checkpoint несовместим с TensorFlow backend")
            weights = [saved[key].copy() for key in keys]
            if any(left.shape != right.shape for left, right in zip(weights, model.get_weights(), strict=False)):
                raise ValueError("Checkpoint model shape несовместим с текущим run")
            model.set_weights(weights)
            if session.data_source is not None and "stream_rows_consumed" in saved:
                session.data_source.restore_rows(int(saved["stream_rows_consumed"]))
