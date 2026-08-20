from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ..metrics import binary_metrics, multiclass_metrics, regression_metrics
from ..models import TaskType
from ..native_core import NativeCoreError, NativeModel, NativeNumericCsvStream
from .base import BackendConfig, EvaluationResult, StepResult, TrainingSession
from .numpy_backend import NumpyBackend


class _NativeStreamingBundle:
    def __init__(self, spec: dict[str, Any]):
        self.train = NativeNumericCsvStream(spec, split="train")
        try:
            validation = NativeNumericCsvStream(spec, split="validation")
            test = NativeNumericCsvStream(spec, split="test")
            self.validation_x, self.validation_y = validation.next_batch(
                min(4096, validation.selected_records)
            )
            self.test_x, self.test_y = test.next_batch(min(4096, test.selected_records))
        except Exception:
            self.train.close()
            raise
        finally:
            if "validation" in locals():
                validation.close()
            if "test" in locals():
                test.close()

    @property
    def train_records(self) -> int:
        return self.train.train_records

    def next_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        return self.train.next_batch(batch_size)

    def state(self) -> dict[str, int]:
        return self.train.state()

    def restore_rows(self, rows_consumed: int) -> None:
        self.train.restore_rows(rows_consumed)

    def close(self) -> None:
        self.train.close()


class NativeBackend(NumpyBackend):
    """Kernelyra's dependency-free C++ training kernel behind the stable backend contract."""

    name = "native"
    version = "1.0"
    export_formats = ("npz", "run-manifest-json", "kernelyra-native-c-abi")

    def create_session(self, config: BackendConfig) -> TrainingSession:
        if config.precision not in {"auto", "float32"}:
            raise ValueError("Native backend supports precision=auto or float32")
        session: TrainingSession | None = None
        if config.dataset_spec:
            try:
                native_source = _NativeStreamingBundle(config.dataset_spec)
            except NativeCoreError:
                # Missing values, categorical features and non-numeric targets
                # remain supported by the general bounded-memory source.
                native_source = None
            if native_source is not None:
                rng = np.random.default_rng(config.seed)
                all_y = np.concatenate((native_source.validation_y, native_source.test_y))
                state: dict[str, Any] = {
                    "task_type": config.task_type,
                    "learning_rate": config.learning_rate
                    or {
                        "eco": .025,
                        "low-memory": .025,
                        "balanced": .035,
                        "performance": .04,
                        "workstation": .04,
                    }.get(config.profile, .035),
                    "weight_decay": float(config.weight_decay),
                    "dtype": np.float32,
                }
                features = native_source.validation_x.shape[1]
                if config.task_type == TaskType.MULTICLASS_CLASSIFICATION.value:
                    class_count = int(max(all_y)) + 1
                    state.update(
                        {
                            "weights": rng.normal(0, .1, (features, class_count)),
                            "bias": np.zeros(class_count, dtype=np.float32),
                            "class_count": class_count,
                        }
                    )
                else:
                    state.update({"weights": rng.normal(0, .1, features), "bias": 0.0})
                if config.task_type == TaskType.REGRESSION.value:
                    state.update(
                        {"target_mean": float(all_y.mean()), "target_std": float(all_y.std()) or 1.0}
                    )
                session = TrainingSession(
                    state,
                    np.empty((0, features), dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                    native_source.validation_x,
                    native_source.validation_y,
                    native_source.test_x,
                    native_source.test_y,
                    rng,
                    metadata={
                        "task_type": config.task_type,
                        "split_seed": config.seed,
                        "test_rows": len(native_source.test_y),
                        "train_records": native_source.train_records,
                        "streaming": True,
                        "precision": "float32",
                        "stream_engine": "kernelyra-native-csv-stream/1",
                    },
                    data_source=native_source,
                )
        if session is None:
            session = super().create_session(config)
        task = str(session.state["task_type"])
        weights = np.asarray(session.state["weights"], dtype=np.float32)
        bias = np.asarray(np.atleast_1d(session.state["bias"]), dtype=np.float32)
        classes = int(session.state.get("class_count", 1))
        model = NativeModel(
            task=task,
            features=int(session.validation_x.shape[1]),
            classes=classes,
            seed=config.seed,
            learning_rate=float(session.state["learning_rate"]),
            weight_decay=float(session.state["weight_decay"]),
            target_mean=float(session.state.get("target_mean", 0.0)),
            target_std=float(session.state.get("target_std", 1.0)),
            threads=max(
                1,
                int(
                    (os.cpu_count() or 1)
                    * int(config.resource_limits.get("cpu_percent", 100))
                    / 100
                ),
            ),
        )
        model.import_parameters(weights, bias)
        session.state["native_model"] = model
        session.metadata.update(
            {
                "backend_version": model.core.version,
                "native_features": model.core.features,
                "engine": "kernelyra-cpp-c-abi",
            }
        )
        if config.dataset_spec and session.metadata.get("stream_engine") and config.checkpoint_path:
            self.restore_checkpoint(session, config.checkpoint_path)
        elif config.dataset_spec and session.metadata.get("stream_engine") and config.model_path:
            if not config.model_path.is_file():
                raise ValueError("Native model file was not found")
            self.restore_checkpoint(session, config.model_path)
        return session

    def train_step(self, session: TrainingSession, batch_size: int) -> StepResult:
        if session.data_source is not None:
            xb, yb = session.data_source.next_batch(batch_size)
            loss = self._model(session).train_step(xb, yb)
        else:
            loss = self._model(session).train_random_step(session.train_x, session.train_y, batch_size)
        if not np.isfinite(loss):
            raise FloatingPointError("NaN guard: native loss became non-finite")
        return StepResult(loss=loss, samples=batch_size)

    def train_steps(self, session: TrainingSession, batch_size: int, steps: int) -> StepResult:
        if not 1 <= steps <= 100:
            raise ValueError("Native train_steps must be between 1 and 100")
        samples = 0
        loss = 0.0
        if session.data_source is None:
            for _ in range(steps):
                loss = self._model(session).train_random_step(
                    session.train_x, session.train_y, batch_size
                )
                samples += batch_size
        else:
            for _ in range(steps):
                xb, yb = session.data_source.next_batch(batch_size)
                loss = self._model(session).train_step(xb, yb)
                samples += len(yb)
        if not np.isfinite(loss):
            raise FloatingPointError("NaN guard: native loss became non-finite")
        return StepResult(loss=loss, samples=samples)

    def evaluate(self, session: TrainingSession) -> EvaluationResult:
        return self._evaluate_native(session, session.validation_x, session.validation_y)

    def evaluate_test(self, session: TrainingSession) -> EvaluationResult:
        return self._evaluate_native(session, session.test_x, session.test_y)

    @staticmethod
    def _model(session: TrainingSession) -> NativeModel:
        model = session.state.get("native_model")
        if not isinstance(model, NativeModel):
            raise RuntimeError("Native model handle is not initialized")
        return model

    def _evaluate_native(self, session: TrainingSession, x: np.ndarray, y: np.ndarray) -> EvaluationResult:
        task = str(session.state["task_type"])
        output = self._model(session).predict(x)
        if task == TaskType.BINARY_CLASSIFICATION.value:
            probabilities = output.reshape(-1)
            bounded = np.clip(probabilities, 1e-8, 1 - 1e-8)
            loss = float(-np.mean(y * np.log(bounded) + (1 - y) * np.log(1 - bounded)))
            metrics = binary_metrics(y, probabilities, loss)
            return EvaluationResult(float(metrics["accuracy"]), metrics)
        if task == TaskType.MULTICLASS_CLASSIFICATION.value:
            truth = y.astype(np.int64)
            loss = float(-np.log(output[np.arange(len(y)), truth] + 1e-8).mean())
            metrics = multiclass_metrics(y, output, loss)
            return EvaluationResult(float(metrics["accuracy"]), metrics)
        metrics = regression_metrics(y, output)
        return EvaluationResult(float(metrics["r2"]), metrics)

    def _sync_parameters(self, session: TrainingSession) -> None:
        weights, bias = self._model(session).export_parameters()
        session.state["weights"] = weights
        session.state["bias"] = float(bias[0]) if len(bias) == 1 else bias

    def save_checkpoint(self, session: TrainingSession, path: Path, metadata: dict[str, Any]) -> None:
        self._sync_parameters(session)
        super().save_checkpoint(session, path, metadata)

    def restore_checkpoint(self, session: TrainingSession, path: Path) -> None:
        super().restore_checkpoint(session, path)
        model = session.state.get("native_model")
        if isinstance(model, NativeModel):
            model.import_parameters(session.state["weights"], session.state["bias"])

    @staticmethod
    def close_session(session: TrainingSession) -> None:
        model = session.state.get("native_model")
        if isinstance(model, NativeModel):
            model.close()
        NumpyBackend.close_session(session)
