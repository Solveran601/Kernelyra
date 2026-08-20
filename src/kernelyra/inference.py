"""Checkpoint-backed inference checks for trained tabular models."""

from __future__ import annotations

import hashlib
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .backends.base import BackendConfig, TrainingSession
from .errors import ConfigurationError, RunError
from .models import RunStatus, TaskType

if TYPE_CHECKING:
    from .workspace import Workspace


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _predict_one(backend_name: str, session: TrainingSession, row: np.ndarray) -> tuple[float | int, float | None]:
    task = str(session.state["task_type"])
    x = row.reshape(1, -1).astype(np.float32, copy=False)
    if backend_name == "torch":
        torch = session.state["torch"]
        model = session.state["model"]
        model.eval()
        with torch.no_grad():
            output = model(torch.as_tensor(x, dtype=torch.float32, device=session.state["device"])).float().cpu().numpy()
        model.train()
    elif backend_name == "tensorflow":
        output = np.asarray(session.state["model"](x, training=False))
    elif backend_name == "native" and "native_model" in session.state:
        output = np.asarray(session.state["native_model"].predict(x))
        if task == TaskType.BINARY_CLASSIFICATION.value:
            probability = float(output.reshape(-1)[0])
            return int(probability >= .5), probability
    else:
        weights = np.asarray(session.state["weights"])
        output = x @ weights + np.asarray(session.state["bias"])

    if task == TaskType.BINARY_CLASSIFICATION.value:
        raw = float(output.reshape(-1)[0])
        probability = raw if backend_name == "tensorflow" else 1.0 / (1.0 + float(np.exp(-np.clip(raw, -30, 30))))
        return int(probability >= .5), probability
    if task == TaskType.MULTICLASS_CLASSIFICATION.value:
        values = output.reshape(1, -1).astype(np.float64)
        if backend_name == "tensorflow":
            probabilities = values
        else:
            values -= values.max(axis=1, keepdims=True)
            exp = np.exp(values)
            probabilities = exp / exp.sum(axis=1, keepdims=True)
        predicted = int(probabilities.argmax(axis=1)[0])
        return predicted, float(probabilities[0, predicted])
    raw = float(output.reshape(-1)[0])
    if backend_name in {"native", "numpy"}:
        raw = raw * float(session.state.get("target_std", 1.0)) + float(session.state.get("target_mean", 0.0))
    return raw, None


def run_inference_check(workspace: Workspace, run_id: str, requests: int = 200) -> dict[str, Any]:
    """Run exactly ``requests`` independent predictions against held-out rows.

    The checkpoint is hashed before and after so the report proves inference did
    not mutate model bytes. This is a tabular inference protocol, not a chat API.
    """
    count = int(requests)
    if not 1 <= count <= 10_000:
        raise ConfigurationError("requests must be between 1 and 10000")
    run = workspace.runs.get(run_id).info
    if run.status != RunStatus.COMPLETED.value:
        raise RunError("Inference check requires a completed run")
    checkpoint = workspace.runtime.checkpoint_path(run.id)
    if not checkpoint.is_file():
        raise RunError("Best checkpoint was not found")
    digest_before = _sha256(checkpoint)
    dataset = workspace.datasets.get(run.dataset)
    dataset_spec = dataset.manifest.get("streaming")
    if dataset_spec:
        x = y = None
    else:
        x, y = workspace.datasets.load_arrays(run.dataset)
    backend_name = run.effective_backend or run.backend
    backend = workspace.backends.create(backend_name)
    total_memory = int(float(workspace.hardware.get("ram_gb") or 8) * 1024**3)
    config = BackendConfig(
        x=x,
        y=y,
        profile=run.profile,
        seed=run.seed,
        task_type=run.objective,
        resource_limits={
            "memory_bytes": max(256 * 1024**2, int(total_memory * run.ram / 100)),
            "cpu_percent": run.cpu,
            "gpu_memory_mb": 0,
        },
        checkpoint_path=checkpoint,
        dataset_spec=dict(dataset_spec) if isinstance(dataset_spec, dict) else None,
        learning_rate=run.learning_rate,
        weight_decay=run.weight_decay,
        hidden_layers=tuple(run.hidden_layers),
        precision=run.precision,
        data_workers=run.data_workers,
        prefetch=0,
    )
    session = backend.create_session(config)
    try:
        available = len(session.test_y)
        if available < count:
            raise RunError(f"Held-out split contains only {available} rows; cannot run {count} independent requests")
        details: list[dict[str, Any]] = []
        latencies: list[float] = []
        correct = 0
        absolute_errors: list[float] = []
        classes = list(dataset.classes)
        for index in range(count):
            started = time.perf_counter_ns()
            predicted, confidence = _predict_one(backend_name, session, session.test_x[index])
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            latencies.append(elapsed_ms)
            expected_raw = session.test_y[index]
            if run.objective == TaskType.REGRESSION.value:
                expected: float | str = float(expected_raw)
                predicted_value: float | str = float(predicted)
                absolute_error = abs(float(predicted) - float(expected_raw))
                absolute_errors.append(absolute_error)
                is_correct = None
            else:
                expected_index = int(expected_raw)
                predicted_index = int(predicted)
                expected = classes[expected_index] if expected_index < len(classes) else str(expected_index)
                predicted_value = classes[predicted_index] if predicted_index < len(classes) else str(predicted_index)
                is_correct = predicted_index == expected_index
                correct += int(is_correct)
            details.append(
                {
                    "request": index + 1,
                    "expected": expected,
                    "predicted": predicted_value,
                    "correct": is_correct,
                    "confidence": confidence,
                    "latency_ms": round(elapsed_ms, 6),
                }
            )
    finally:
        backend.close_session(session)
    digest_after = _sha256(checkpoint)
    ordered = sorted(latencies)
    percentile_index = min(len(ordered) - 1, max(0, int(round(.95 * len(ordered) + .5)) - 1))
    summary: dict[str, Any] = {
        "requests": count,
        "total_ms": round(sum(latencies), 6),
        "mean_ms": round(statistics.fmean(latencies), 6),
        "median_ms": round(statistics.median(latencies), 6),
        "p95_ms": round(ordered[percentile_index], 6),
        "requests_per_second": round(1000.0 * count / max(sum(latencies), 1e-9), 3),
    }
    if run.objective == TaskType.REGRESSION.value:
        summary["mean_absolute_error"] = float(statistics.fmean(absolute_errors))
    else:
        summary["correct"] = correct
        summary["accuracy"] = correct / count
    return {
        "contract": "kernelyra-inference-check/1",
        "run_id": run.id,
        "backend": backend_name,
        "architecture": run.architecture,
        "model_format": run.model_format,
        "task": run.objective,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": digest_before,
        "checkpoint_sha256_after": digest_after,
        "checkpoint_immutable": digest_before == digest_after,
        "protocol": "structured-tabular-prediction",
        "chat_model": False,
        "summary": summary,
        "results": details,
    }
