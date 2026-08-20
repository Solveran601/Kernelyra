from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ..metrics import binary_metrics, multiclass_metrics, regression_metrics
from ..models import TaskType
from ..streaming import StreamingTabularSource
from .base import BackendConfig, EvaluationResult, StepResult, TrainingSession


def _torch() -> Any:
    import torch

    return torch


class TorchBackend:
    name = "torch"
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
    export_formats = ("pt-state-dict", "npz-checkpoint", "run-manifest-json")

    def inspect_model(self, path: Path) -> dict[str, Any]:
        valid = path.is_file() and path.suffix.lower() in {".pt", ".pth"}
        return {"backend": self.name, "valid": valid, "fine_tune": valid}

    def create_session(self, config: BackendConfig) -> TrainingSession:
        torch = _torch()
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        threads = max(1, int((os.cpu_count() or 1) * int(config.resource_limits.get("cpu_percent", 50)) / 100))
        torch.set_num_threads(threads)
        use_cuda = bool(config.resource_limits.get("gpu_enabled")) and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda:
            gpu_limit = int(config.resource_limits.get("gpu_memory_mb") or 0)
            if gpu_limit > 0:
                fraction = min(1.0, max(.05, gpu_limit / max(1, torch.cuda.get_device_properties(0).total_memory / 1024**2)))
                torch.cuda.set_per_process_memory_fraction(fraction, 0)

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
                raise ValueError("PyTorch backend requires arrays or a streaming dataset spec")
            x, y = config.x.astype(np.float32), config.y
            order = rng.permutation(len(x))
            x, y = x[order], y[order]
            validation_size = max(16, int(round(len(x) * config.validation_fraction)))
            test_size = max(1, int(round(len(x) * config.test_fraction)))
            train_end = len(x) - validation_size - test_size
            if train_end < 8:
                raise ValueError("Not enough rows after deterministic train/validation/test split")
            train_x, validation_x = x[:train_end], x[train_end : train_end + validation_size]
            train_y, validation_y = y[:train_end], y[train_end : train_end + validation_size]
            test_x, test_y = x[train_end + validation_size :], y[train_end + validation_size :]
            all_y = y
        test_size = len(test_y)
        class_count = int(max(all_y)) + 1 if config.task_type == TaskType.MULTICLASS_CLASSIFICATION.value else 1
        widths = list(config.hidden_layers or (64, 32))
        layers: list[Any] = []
        input_width = validation_x.shape[1]
        for width in widths:
            layers.extend((torch.nn.Linear(input_width, int(width)), torch.nn.ReLU()))
            input_width = int(width)
        layers.append(torch.nn.Linear(input_width, class_count))
        model = torch.nn.Sequential(*layers).to(device)
        if config.model_path:
            try:
                payload = torch.load(config.model_path, map_location=device, weights_only=True)
            except TypeError as error:
                raise ValueError("This PyTorch version lacks safe weights_only model loading") from error
            state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            if not isinstance(state_dict, dict):
                raise ValueError("PyTorch model must contain a state_dict")
            model.load_state_dict(state_dict, strict=True)

        learning_rate = config.learning_rate
        if learning_rate is None:
            learning_rate = {
                "eco": .0008,
                "low-memory": .0008,
                "balanced": .001,
                "performance": .0015,
                "workstation": .0015,
            }.get(config.profile, .001)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=config.weight_decay)
        requested_precision = config.precision
        if requested_precision not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("PyTorch precision must be auto, float32, float16 or bfloat16")
        precision = "float16" if requested_precision == "auto" and use_cuda else (
            "float32" if requested_precision == "auto" else requested_precision
        )
        if precision == "float16" and not use_cuda:
            raise ValueError("PyTorch float16 training requires CUDA; use float32 or bfloat16 on CPU")
        if precision == "bfloat16" and use_cuda and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bfloat16")
        amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(precision)
        scaler = None
        if precision == "float16":
            try:
                scaler = torch.amp.GradScaler("cuda")
            except (AttributeError, TypeError):
                scaler = torch.cuda.amp.GradScaler()
        if config.task_type == TaskType.BINARY_CLASSIFICATION.value:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        elif config.task_type == TaskType.MULTICLASS_CLASSIFICATION.value:
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            loss_fn = torch.nn.MSELoss()
        state = {
            "torch": torch,
            "model": model,
            "optimizer": optimizer,
            "loss_fn": loss_fn,
            "task_type": config.task_type,
            "device": device,
            "class_count": class_count,
            "precision": precision,
            "amp_dtype": amp_dtype,
            "scaler": scaler,
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
                "device": str(device),
                "cpu_threads": threads,
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
        torch = session.state["torch"]
        device = session.state["device"]
        if session.data_source is not None:
            batch_x, batch_y = session.data_source.next_batch(batch_size)
        else:
            indices = session.rng.integers(0, len(session.train_x), size=batch_size)
            batch_x, batch_y = session.train_x[indices], session.train_y[indices]
        xb = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
        task = session.state["task_type"]
        if task == TaskType.MULTICLASS_CLASSIFICATION.value:
            yb = torch.as_tensor(batch_y, dtype=torch.long, device=device)
        else:
            yb = torch.as_tensor(batch_y, dtype=torch.float32, device=device).reshape(-1, 1)
        model, optimizer = session.state["model"], session.state["optimizer"]
        optimizer.zero_grad(set_to_none=True)
        amp_dtype = session.state["amp_dtype"]
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            output = model(xb)
            loss = session.state["loss_fn"](output, yb)
        scaler = session.state["scaler"]
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        value = float(loss.detach().cpu().item())
        if not np.isfinite(value):
            raise FloatingPointError("NaN guard: loss is not finite")
        return StepResult(loss=value, samples=batch_size)

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
        torch = session.state["torch"]
        model = session.state["model"]
        model.eval()
        with torch.no_grad():
            device = session.state["device"]
            with torch.autocast(
                device_type=device.type,
                dtype=session.state["amp_dtype"],
                enabled=session.state["amp_dtype"] is not None,
            ):
                output = model(torch.as_tensor(x, dtype=torch.float32, device=device)).float().cpu().numpy()
        model.train()
        task = session.state["task_type"]
        if task == TaskType.BINARY_CLASSIFICATION.value:
            probabilities = 1 / (1 + np.exp(-np.clip(output.reshape(-1), -30, 30)))
            loss = float(-np.mean(y * np.log(probabilities + 1e-8) + (1 - y) * np.log(1 - probabilities + 1e-8)))
            metrics = binary_metrics(y, probabilities, loss)
            return EvaluationResult(float(metrics["accuracy"]), metrics)
        if task == TaskType.MULTICLASS_CLASSIFICATION.value:
            logits = output - output.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probabilities = exp / exp.sum(axis=1, keepdims=True)
            truth = y.astype(np.int64)
            loss = float(-np.log(probabilities[np.arange(len(y)), truth] + 1e-8).mean())
            metrics = multiclass_metrics(y, probabilities, loss)
            return EvaluationResult(float(metrics["accuracy"]), metrics)
        metrics = regression_metrics(y, output.reshape(-1))
        return EvaluationResult(float(metrics["r2"]), metrics)

    def save_checkpoint(self, session: TrainingSession, path: Path, metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(path.stem + ".pending.npz")
        parameters = {
            f"param::{name}": value.detach().cpu().numpy()
            for name, value in session.state["model"].state_dict().items()
        }
        stream_state = session.data_source.state() if session.data_source is not None else {}
        payload = {
            **metadata,
            **stream_state,
            "task_type": session.state["task_type"],
            **parameters,
        }
        np.savez(pending, **payload)
        os.replace(pending, path)

    def restore_checkpoint(self, session: TrainingSession, path: Path) -> None:
        torch = session.state["torch"]
        model = session.state["model"]
        with np.load(path, allow_pickle=False) as saved:
            if "task_type" in saved and str(saved["task_type"]) != session.state["task_type"]:
                raise ValueError("Checkpoint task_type is incompatible with the current run")
            current = model.state_dict()
            restored = {}
            for name, value in current.items():
                key = f"param::{name}"
                if key not in saved or tuple(saved[key].shape) != tuple(value.shape):
                    raise ValueError("Checkpoint model shape is incompatible with PyTorch backend")
                restored[name] = torch.as_tensor(saved[key], dtype=value.dtype, device=session.state["device"])
            model.load_state_dict(restored, strict=True)
            if session.data_source is not None and "stream_rows_consumed" in saved:
                session.data_source.restore_rows(int(saved["stream_rows_consumed"]))
