from __future__ import annotations

import json
import multiprocessing as mp
import os
import secrets
import signal
import socket
import struct
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from ..errors import WorkerCrashedError, WorkerProtocolError, WorkerTimeoutError
from ..resource_control import WindowsJob, apply_child_limits
from .base import BackendConfig, EvaluationResult, StepResult, TrainingBackend, TrainingSession

WORKER_PROTOCOL_VERSION = "kernelyra-worker/2"
MAX_IPC_MESSAGE_BYTES = 1024 * 1024


class BackendWorker(Protocol):
    protocol_version: str
    effective_backend: str
    worker_pid: int | None
    resource_enforcement: dict[str, Any]

    @property
    def train_records(self) -> int: ...
    def train_step(self, batch_size: int) -> StepResult: ...
    def train_steps(self, batch_size: int, steps: int) -> StepResult: ...
    def evaluate(self) -> EvaluationResult: ...
    def evaluate_test(self) -> EvaluationResult: ...
    def restore_checkpoint(self, path: Path) -> None: ...
    def save_checkpoint(self, path: Path, metadata: dict[str, Any]) -> None: ...
    def drain_events(self) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


class InProcessBackendWorker:
    """Embedded/test-only adapter. Production runtime uses ProcessBackendWorker."""

    protocol_version = WORKER_PROTOCOL_VERSION

    def __init__(self, backend: TrainingBackend, session: TrainingSession):
        self.backend = backend
        self._session = session
        self.effective_backend = str(getattr(backend, "name", "unknown"))
        self.worker_pid = os.getpid()
        self.resource_enforcement = {
            "requested": {},
            "scheduler_enforced": True,
            "os_enforced": {},
            "backend_enforced": {},
            "unsupported": ["embedded_mode_has_no_process_boundary"],
            "degraded": [],
        }

    @property
    def session(self) -> TrainingSession:
        return self._session

    @property
    def train_records(self) -> int:
        return int(self._session.metadata.get("train_records") or len(self._session.train_x))

    def train_step(self, batch_size: int) -> StepResult:
        return self.backend.train_step(self._session, batch_size)

    def train_steps(self, batch_size: int, steps: int) -> StepResult:
        backend_train_steps = getattr(self.backend, "train_steps", None)
        if callable(backend_train_steps):
            accelerated = backend_train_steps(self._session, batch_size, steps)
            if not isinstance(accelerated, StepResult):
                raise WorkerProtocolError("Backend train_steps must return StepResult")
            return accelerated
        result = StepResult(loss=0.0, samples=0)
        for _ in range(steps):
            step = self.backend.train_step(self._session, batch_size)
            result = StepResult(loss=step.loss, samples=result.samples + step.samples)
        return result

    def evaluate(self) -> EvaluationResult:
        return self.backend.evaluate(self._session)

    def evaluate_test(self) -> EvaluationResult:
        return self.backend.evaluate_test(self._session)

    def restore_checkpoint(self, path: Path) -> None:
        self.backend.restore_checkpoint(self._session, path)

    def save_checkpoint(self, path: Path, metadata: dict[str, Any]) -> None:
        self.backend.save_checkpoint(self._session, path, metadata)

    def drain_events(self) -> list[dict[str, Any]]:
        return []

    def close(self) -> None:
        closer = getattr(self.backend, "close_session", None)
        if callable(closer):
            closer(self._session)


def _receive_exact(connection: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("Worker IPC socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_message(connection: socket.socket, message: dict[str, Any]) -> None:
    payload = json.dumps(message, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(payload) > MAX_IPC_MESSAGE_BYTES:
        raise WorkerProtocolError("Worker IPC message exceeds 1 MB")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def _receive_message(connection: socket.socket) -> dict[str, Any]:
    length = struct.unpack("!I", _receive_exact(connection, 4))[0]
    if length <= 0 or length > MAX_IPC_MESSAGE_BYTES:
        raise WorkerProtocolError("Invalid worker IPC frame length")
    value = json.loads(_receive_exact(connection, length).decode("utf-8"))
    if not isinstance(value, dict):
        raise WorkerProtocolError("Worker IPC frame must be a JSON object")
    return value


def _event(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": WORKER_PROTOCOL_VERSION,
        "type": event_type,
        "payload": payload,
        "created_at": time.time(),
    }


def _worker_main(
    host: str,
    port: int,
    bootstrap_token: str,
    heartbeat: Any,
    backend_name: str,
    config: BackendConfig,
    allow_numpy_fallback: bool,
) -> None:
    sys.dont_write_bytecode = True
    connection: socket.socket | None = None
    stop_heartbeat = threading.Event()

    def beat() -> None:
        while not stop_heartbeat.wait(.5):
            heartbeat.value = time.monotonic()

    heartbeat_thread = threading.Thread(target=beat, name="kernelyra-worker-heartbeat", daemon=True)
    heartbeat_thread.start()
    try:
        connection = socket.create_connection((host, port), timeout=10)
        connection.settimeout(None)
        enforcement = apply_child_limits(config.resource_limits)
        from .registry import BackendRegistry

        registry = BackendRegistry()
        effective_backend = backend_name
        startup_events: list[dict[str, Any]] = []
        try:
            backend = registry.create(effective_backend)
            session = backend.create_session(config)
        except ModuleNotFoundError as error:
            optional_backend = (effective_backend == "tensorflow" and error.name == "tensorflow") or (
                effective_backend == "torch" and error.name == "torch"
            )
            if not allow_numpy_fallback or not optional_backend:
                raise
            effective_backend = "numpy"
            backend = registry.create(effective_backend)
            session = backend.create_session(config)
            startup_events.append(
                _event("log", {"level": "warning", "message": f"{backend_name} unavailable; NumPy fallback selected"})
            )
        if int(config.resource_limits.get("gpu_memory_mb") or 0):
            if effective_backend in {"tensorflow", "torch"}:
                enforcement["backend_enforced"]["gpu_memory"] = f"{effective_backend}_configured"
            else:
                enforcement["backend_enforced"]["gpu_memory"] = "unsupported"
                enforcement["unsupported"].append("gpu_memory_limit_for_backend")
        _send_message(
            connection,
            {
                "protocol": WORKER_PROTOCOL_VERSION,
                "type": "ready",
                "bootstrap_token": bootstrap_token,
                "effective_backend": effective_backend,
                "pid": os.getpid(),
                "train_records": int(session.metadata.get("train_records", len(session.train_x))),
                "metadata": session.metadata,
                "resource_enforcement": enforcement,
                "events": startup_events + [_event("worker.ready", {"pid": os.getpid(), "backend": effective_backend})],
            },
        )
        while True:
            message = _receive_message(connection)
            request_id = str(message.get("id", ""))
            response_events: list[dict[str, Any]] = []
            if message.get("protocol") != WORKER_PROTOCOL_VERSION:
                _send_message(
                    connection,
                    {
                        "protocol": WORKER_PROTOCOL_VERSION,
                        "id": request_id,
                        "type": "error",
                        "error_type": "WorkerProtocolError",
                        "error": "Incompatible worker protocol",
                        "events": response_events,
                    },
                )
                continue
            command = message.get("command")
            payload = message.get("payload") or {}
            try:
                if command == "train_step":
                    result = backend.train_step(session, int(payload["batch_size"]))
                    response = asdict(result)
                    response_events.append(_event("progress", response))
                elif command == "train_steps":
                    count = max(1, min(100, int(payload["steps"])))
                    batch_size = int(payload["batch_size"])
                    backend_train_steps = getattr(backend, "train_steps", None)
                    if callable(backend_train_steps):
                        result = backend_train_steps(session, batch_size, count)
                    else:
                        result = StepResult(loss=0.0, samples=0)
                        for _ in range(count):
                            current = backend.train_step(session, batch_size)
                            result = StepResult(
                                loss=current.loss, samples=result.samples + current.samples
                            )
                    response = {**asdict(result), "steps": count}
                    response_events.append(_event("progress", response))
                elif command == "evaluate":
                    result = backend.evaluate(session)
                    response = asdict(result)
                    response_events.append(_event("metrics", response))
                elif command == "evaluate_test":
                    result = backend.evaluate_test(session)
                    response = asdict(result)
                    response_events.append(_event("metrics.test", response))
                elif command == "restore_checkpoint":
                    backend.restore_checkpoint(session, Path(payload["path"]))
                    response = {"restored": True}
                    response_events.append(
                        _event("checkpoint.restored", {"path_name": Path(payload["path"]).name})
                    )
                elif command == "save_checkpoint":
                    backend.save_checkpoint(session, Path(payload["path"]), dict(payload["metadata"]))
                    response = {"saved": True}
                    response_events.append(
                        _event("checkpoint", {"path_name": Path(payload["path"]).name})
                    )
                elif command == "close":
                    closer = getattr(backend, "close_session", None)
                    if callable(closer):
                        closer(session)
                    _send_message(
                        connection,
                        {
                            "protocol": WORKER_PROTOCOL_VERSION,
                            "id": request_id,
                            "type": "result",
                            "payload": {"closed": True},
                            "events": [_event("worker.closed", {"pid": os.getpid()})],
                        },
                    )
                    break
                else:
                    raise WorkerProtocolError(f"Unknown worker command: {command}")
                _send_message(
                    connection,
                    {
                        "protocol": WORKER_PROTOCOL_VERSION,
                        "id": request_id,
                        "type": "result",
                        "payload": response,
                        "events": response_events,
                    },
                )
            except Exception as error:
                _send_message(
                    connection,
                    {
                        "protocol": WORKER_PROTOCOL_VERSION,
                        "id": request_id,
                        "type": "error",
                        "error_type": type(error).__name__,
                        "error": str(error)[:500],
                        "events": response_events,
                    },
                )
    except Exception as error:
        if connection is not None:
            try:
                _send_message(
                    connection,
                    {
                        "protocol": WORKER_PROTOCOL_VERSION,
                        "type": "startup_error",
                        "bootstrap_token": bootstrap_token,
                        "error_type": type(error).__name__,
                        "error": str(error)[:500],
                        "events": [],
                    },
                )
            except (BrokenPipeError, EOFError, OSError):
                pass
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1)
        if connection is not None:
            connection.close()


@dataclass(slots=True)
class ProcessWorkerStatus:
    forced_termination: bool = False
    close_reason: str = "not_closed"


class ProcessBackendWorker:
    """Spawned backend worker using authenticated, versioned loopback JSON IPC."""

    protocol_version = WORKER_PROTOCOL_VERSION

    def __init__(
        self,
        backend_name: str,
        config: BackendConfig,
        *,
        allow_numpy_fallback: bool = True,
        request_timeout: float = 30.0,
        startup_timeout: float = 120.0,
        close_timeout: float = 5.0,
    ):
        self.request_timeout = request_timeout
        self.startup_timeout = startup_timeout
        self.close_timeout = close_timeout
        self.status = ProcessWorkerStatus()
        self._closed = False
        self._event_buffer: list[dict[str, Any]] = []
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(startup_timeout)
        host, port = listener.getsockname()
        bootstrap_token = secrets.token_urlsafe(32)
        context = mp.get_context("spawn")
        self._heartbeat: Any = context.RawValue("d", time.monotonic())
        self._process = context.Process(
            target=_worker_main,
            args=(
                host,
                port,
                bootstrap_token,
                self._heartbeat,
                backend_name,
                config,
                allow_numpy_fallback,
            ),
            name=f"kernelyra-backend-{backend_name}",
            daemon=False,
        )
        try:
            self._process.start()
            child_pid = self._process.pid
            if child_pid is None:
                raise WorkerCrashedError("Backend worker did not receive a process id")
            self._windows_job = WindowsJob(child_pid, config.resource_limits)
            self._connection, peer = listener.accept()
            if peer[0] != "127.0.0.1":
                raise WorkerProtocolError("Worker IPC accepted a non-loopback peer")
            self._connection.settimeout(request_timeout)
            ready = _receive_message(self._connection)
        except Exception as error:
            listener.close()
            if hasattr(self, "_process") and self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
            if isinstance(error, TimeoutError):
                raise WorkerTimeoutError("Backend worker timed out during startup") from None
            raise
        finally:
            listener.close()
        if ready.get("bootstrap_token") != bootstrap_token:
            self._terminate_tree("bootstrap_auth_failed")
            raise WorkerProtocolError("Backend worker bootstrap authentication failed")
        if ready.get("protocol") != WORKER_PROTOCOL_VERSION:
            self._terminate_tree("protocol_mismatch")
            raise WorkerProtocolError("Backend worker protocol is incompatible")
        if ready.get("type") != "ready":
            self._terminate_tree("startup_error")
            raise WorkerCrashedError(
                f"Worker startup failed: {ready.get('error_type', 'Error')}: {ready.get('error', '')}"
            )
        self._event_buffer.extend(ready.get("events") or [])
        self.effective_backend = str(ready["effective_backend"])
        self.worker_pid: int | None = int(ready["pid"])
        self._train_records = int(ready["train_records"])
        self.metadata = dict(ready.get("metadata") or {})
        child_enforcement = dict(ready.get("resource_enforcement") or {})
        if os.name == "nt":
            self.resource_enforcement = {
                **self._windows_job.status,
                "backend_enforced": child_enforcement.get("backend_enforced", {}),
                "unsupported": sorted(
                    set(self._windows_job.status.get("unsupported", []))
                    | set(child_enforcement.get("unsupported", []))
                ),
                "degraded": [
                    *self._windows_job.status.get("degraded", []),
                    *child_enforcement.get("degraded", []),
                ],
            }
        else:
            self.resource_enforcement = child_enforcement

    @property
    def train_records(self) -> int:
        return self._train_records

    @property
    def heartbeat_age(self) -> float:
        return max(0.0, time.monotonic() - float(self._heartbeat.value))

    def _request(self, command: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if self._closed or not self._process.is_alive():
            raise WorkerCrashedError(f"Backend worker exited with code {self._process.exitcode}")
        if self.heartbeat_age > max(5.0, self.request_timeout):
            self._terminate_tree("heartbeat_timeout")
            raise WorkerTimeoutError("Backend worker heartbeat expired")
        request_id = uuid.uuid4().hex
        self._connection.settimeout(self.request_timeout if timeout is None else timeout)
        try:
            _send_message(
                self._connection,
                {
                    "protocol": WORKER_PROTOCOL_VERSION,
                    "id": request_id,
                    "command": command,
                    "payload": payload,
                },
            )
            response = _receive_message(self._connection)
        except TimeoutError:
            self._terminate_tree(f"{command}_timeout")
            raise WorkerTimeoutError(f"Backend worker timed out during {command}") from None
        except (BrokenPipeError, ConnectionError, EOFError, OSError) as error:
            self._terminate_tree(f"{command}_ipc_failure")
            raise WorkerCrashedError(f"Backend worker IPC failed: {type(error).__name__}") from None
        self._event_buffer.extend(response.get("events") or [])
        if response.get("protocol") != WORKER_PROTOCOL_VERSION or response.get("id") != request_id:
            self._terminate_tree("response_protocol_mismatch")
            raise WorkerProtocolError("Backend worker returned an incompatible response")
        if response.get("type") == "error":
            raise WorkerCrashedError(
                f"{response.get('error_type', 'WorkerError')}: {response.get('error', '')}"
            )
        return dict(response.get("payload") or {})

    def train_step(self, batch_size: int) -> StepResult:
        return StepResult(**self._request("train_step", {"batch_size": batch_size}))

    def train_steps(self, batch_size: int, steps: int) -> StepResult:
        payload = self._request("train_steps", {"batch_size": batch_size, "steps": steps})
        executed_steps = int(payload.pop("steps", 0))
        if executed_steps != steps:
            self._terminate_tree("train_steps_count_mismatch")
            raise WorkerProtocolError(
                f"Backend worker executed {executed_steps} steps instead of {steps}"
            )
        return StepResult(**payload)

    def evaluate(self) -> EvaluationResult:
        return EvaluationResult(**self._request("evaluate", {}))

    def evaluate_test(self) -> EvaluationResult:
        return EvaluationResult(**self._request("evaluate_test", {}))

    def restore_checkpoint(self, path: Path) -> None:
        self._request("restore_checkpoint", {"path": str(path)})

    def save_checkpoint(self, path: Path, metadata: dict[str, Any]) -> None:
        self._request("save_checkpoint", {"path": str(path), "metadata": metadata})

    def drain_events(self) -> list[dict[str, Any]]:
        events = self._event_buffer[:256]
        del self._event_buffer[: len(events)]
        return events

    def _terminate_tree(self, reason: str) -> None:
        self.status.forced_termination = True
        self.status.close_reason = reason
        if os.name == "nt" and hasattr(self, "_windows_job"):
            self._windows_job.terminate(1)
        elif self._process.pid and self._process.pid != os.getpid():
            try:
                getattr(os, "killpg")(self._process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                self._process.terminate()
        self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=2)

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.is_alive():
                self._request("close", {}, timeout=self.close_timeout)
                self.status.close_reason = "cooperative"
                self._process.join(timeout=self.close_timeout)
            if self._process.is_alive():
                self._terminate_tree("close_timeout")
        except (WorkerCrashedError, WorkerTimeoutError, EOFError, BrokenPipeError):
            if self._process.is_alive():
                self._terminate_tree("close_failure")
        finally:
            self._closed = True
            if hasattr(self, "_connection"):
                self._connection.close()
            if hasattr(self, "_windows_job"):
                self._windows_job.close()
