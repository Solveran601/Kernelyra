from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import secrets
import sys
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .agent_policy import AgentPolicy
from .architectures import CHECKPOINT_FORMATS
from .errors import (
    AccessDeniedError,
    ApprovalError,
    KernelyraError,
    NotFoundError,
    PayloadTooLargeError,
    RateLimitError,
    RunStateError,
)
from .formats import describe_formats
from .models import DatasetInfo, RunConfig, RunInfo
from .security import LOOPBACK_HOSTS, ensure_agent_secret, ensure_user_secret, validate_daemon_bind
from .workspace import Workspace, batch_plan_for

VERSION = "0.4.0a1"


class BatchPlanRequest(BaseModel):
    dataset: str = "demo"
    profile: str = "auto"
    batch_mode: str = "auto"
    batch_size: int | None = None
    ram: int = Field(default=35, ge=10, le=95)


class RunRequest(BaseModel):
    name: str = "new-classifier"
    dataset: str = "demo"
    backend: str = "tensorflow"
    objective: str = "binary_classification"
    architecture: str = "auto"
    model_format: str = "auto"
    mode: str = "Новая модель"
    profile: str = "auto"
    priority: str = "normal"
    target_score: float = .92
    batch_mode: str = "auto"
    batch_size: int | None = None
    accept_batch_risk: bool = False
    max_steps: int = 1400
    cpu: int | None = None
    ram: int | None = None
    gpu: int | None = None
    base_run_id: str | None = None
    model_path: str | None = None
    seed: int = 42
    learning_rate: float | None = None
    weight_decay: float = 0.0
    hidden_layers: tuple[int, ...] = ()
    precision: str = "auto"
    data_workers: int = 0
    prefetch: int = 1
    evaluation_interval: int | None = Field(default=None, ge=1, le=1_000_000)
    min_improvement: float = Field(default=.0005, ge=0, le=1)
    degradation_margin: float | None = Field(default=None, gt=0, le=10)
    degradation_patience: int = Field(default=3, ge=1, le=100)
    early_stopping_patience: int = Field(default=18, ge=1, le=10_000)
    target_patience: int = Field(default=3, ge=1, le=100)
    start: bool = True


class PathRequest(BaseModel):
    path: str
    target: str | None = None


class CommandRequest(BaseModel):
    command: str


class MCPCommandRequest(CommandRequest):
    approval_token: str | None = None


class ApprovalRequest(BaseModel):
    action: str
    resource_id: str
    ttl_seconds: int = Field(default=300, ge=30, le=3600)


class MCPDatasetRequest(PathRequest):
    approval_token: str


class AgentSessionRequest(BaseModel):
    client_id: str = Field(min_length=8, max_length=160)
    ttl_seconds: int = Field(default=300, ge=60, le=600)


class TokenRequest(BaseModel):
    token: str = Field(min_length=16, max_length=512)


class MCPExportRequest(BaseModel):
    approval_token: str = Field(min_length=16, max_length=512)


class RunMetricsResponse(BaseModel):
    run_id: str
    step: int
    metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ScopedAgentSession:
    session_id: str
    client_id: str
    expires_at: float
    policy: AgentPolicy


def create_app(
    workspace_path: str | Path,
    assets_root: str | Path | None = None,
    api_token: str | None = None,
    *,
    enable_ui: bool = False,
    allowed_hosts: list[str] | None = None,
) -> FastAPI:
    workspace = Workspace.open(workspace_path)
    user_secret = ensure_user_secret(workspace.state_dir)
    agent_secret = ensure_agent_secret(workspace.state_dir)
    _ = assets_root, enable_ui  # Retained only for API compatibility; Kernelyra is headless.
    agent_sessions: dict[str, ScopedAgentSession] = {}
    rate_windows: dict[str, deque[float]] = defaultdict(deque)
    approved_hosts = {item.strip().lower() for item in (allowed_hosts or []) if item.strip()}
    approved_hosts.update({"127.0.0.1", "localhost", "::1", "testserver"})

    def limited(bucket: str, request: Request, maximum: int, window_seconds: int = 60) -> bool:
        address = request.client.host if request.client else "unknown"
        key = f"{bucket}:{address}"
        now = time.monotonic()
        window = rate_windows[key]
        while window and window[0] <= now - window_seconds:
            window.popleft()
        if len(window) >= maximum:
            return True
        window.append(now)
        return False

    def has_user_access(request: Request) -> bool:
        supplied_secret = request.headers.get("X-Kernelyra-User-Secret", "")
        if supplied_secret and secrets.compare_digest(supplied_secret, user_secret):
            return True
        if api_token:
            supplied_token = request.headers.get("Authorization", "")
            if secrets.compare_digest(supplied_token, f"Bearer {api_token}"):
                return True
        return False

    def has_agent_bootstrap_access(request: Request) -> bool:
        supplied = request.headers.get("X-Kernelyra-Agent-Secret", "")
        return bool(supplied) and secrets.compare_digest(supplied, agent_secret)

    def get_agent_session(request: Request) -> ScopedAgentSession | None:
        token = request.headers.get("X-Kernelyra-Agent-Session", "")
        session = agent_sessions.get(token)
        if session is None:
            return None
        if session.expires_at <= time.time():
            agent_sessions.pop(token, None)
            return None
        supplied_client = request.headers.get("X-Kernelyra-Agent-Client", "")
        if not supplied_client or not secrets.compare_digest(supplied_client, session.client_id):
            return None
        return session

    def require_agent_action(request: Request, action: str) -> ScopedAgentSession:
        session = get_agent_session(request)
        if session is None:
            raise AccessDeniedError("Нужна действующая scoped agent session")
        session.policy.require_action(action)
        workspace.storage.log_action(
            "mcp_agent",
            action,
            {
                "session_id": session.session_id,
                "client_id": session.client_id,
                "method": request.method,
                "route": request.url.path,
            },
        )
        return session

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        workspace.runtime.start()
        try:
            yield
        finally:
            workspace.close()

    app = FastAPI(
        title="Kernelyra API",
        version=VERSION,
        docs_url="/api/docs",
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan,
    )
    app.state.workspace = workspace
    @app.middleware("http")
    async def api_authentication(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        supplied_request_id = request.headers.get("X-Request-ID", "")
        request_id = (
            supplied_request_id
            if re.fullmatch(r"[A-Za-z0-9._-]{8,80}", supplied_request_id)
            else uuid.uuid4().hex
        )
        request.state.request_id = request_id

        def finish(response: Response) -> Response:
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Kernelyra-API-Version"] = "v1"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
            response.headers.setdefault("Cache-Control", "no-store")
            return response

        def denied(status: int, message: str, error_type: str) -> Response:
            return finish(
                JSONResponse(
                    status_code=status,
                    content={"error": message, "type": error_type, "request_id": request_id},
                )
            )

        host_header = request.headers.get("host", "")
        try:
            host_name = urlparse("http://" + host_header).hostname or ""
        except ValueError:
            host_name = ""
        if host_name.lower() not in approved_hosts:
            return denied(400, "Host header is not approved", "HostValidationError")

        path = request.url.path
        if not path.startswith("/api") or path in {"/api/health", "/api/v1/health"}:
            return finish(await call_next(request))
        if api_token:
            supplied = request.headers.get("Authorization", "")
            expected = f"Bearer {api_token}"
            if not secrets.compare_digest(supplied, expected):
                return denied(401, "Неверный API token", "AuthenticationError")
        if path.startswith("/api/v1/mcp/"):
            if path == "/api/v1/mcp/sessions" and has_agent_bootstrap_access(request):
                if limited("agent-session", request, 20):
                    return denied(429, "Agent session rate limit exceeded", "RateLimitError")
                return finish(await call_next(request))
            if get_agent_session(request) is not None:
                return finish(await call_next(request))
            return denied(403, "Нужна действующая scoped agent session", "AgentSessionRequired")
        if not has_user_access(request):
            return denied(403, "Нужна защищённая локальная пользовательская сессия", "UserSessionRequired")
        if path == "/api/v1/approvals" and limited("approval", request, 30):
            return denied(429, "Approval rate limit exceeded", "RateLimitError")
        return finish(await call_next(request))

    @app.exception_handler(RunStateError)
    async def run_state_error(request: Request, error: RunStateError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "error": str(error),
                "type": type(error).__name__,
                "run_id": error.run_id,
                "command": error.command,
                "status": error.status,
                "allowed": list(error.allowed),
                "request_id": request.state.request_id,
            },
        )

    @app.exception_handler(ApprovalError)
    async def approval_error(request: Request, error: ApprovalError) -> JSONResponse:
        return JSONResponse(status_code=403, content={"error": str(error), "type": type(error).__name__, "request_id": request.state.request_id})

    @app.exception_handler(AccessDeniedError)
    async def access_denied_error(request: Request, error: AccessDeniedError) -> JSONResponse:
        return JSONResponse(status_code=403, content={"error": str(error), "type": type(error).__name__, "request_id": request.state.request_id})

    @app.exception_handler(NotFoundError)
    async def not_found_error(request: Request, error: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": str(error), "type": type(error).__name__, "request_id": request.state.request_id})

    @app.exception_handler(PayloadTooLargeError)
    async def payload_too_large_error(request: Request, error: PayloadTooLargeError) -> JSONResponse:
        return JSONResponse(status_code=413, content={"error": str(error), "type": type(error).__name__, "request_id": request.state.request_id})

    @app.exception_handler(RateLimitError)
    async def rate_limit_error(request: Request, error: RateLimitError) -> JSONResponse:
        return JSONResponse(status_code=429, content={"error": str(error), "type": type(error).__name__, "request_id": request.state.request_id})

    @app.exception_handler(KernelyraError)
    async def kernelyra_error(request: Request, error: KernelyraError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(error), "type": type(error).__name__, "request_id": request.state.request_id})

    @app.exception_handler(ValueError)
    async def value_error(request: Request, error: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(error), "type": type(error).__name__, "request_id": request.state.request_id})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        message = error.errors()[0].get("msg", "Некорректный запрос") if error.errors() else "Некорректный запрос"
        return JSONResponse(status_code=422, content={"error": message, "type": "RequestValidationError", "details": error.errors(), "request_id": request.state.request_id})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
        error_type = "NotFoundError" if error.status_code == 404 else "HTTPError"
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": str(error.detail),
                "type": error_type,
                "request_id": request.state.request_id,
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        workspace.storage.log_action(
            "server",
            "request.error",
            {"request_id": request.state.request_id, "error_type": type(error).__name__},
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "type": "InternalServerError",
                "request_id": request.state.request_id,
            },
        )

    @app.get("/api/health")
    @app.get("/api/v1/health")
    def health(request: Request) -> dict[str, Any]:
        public = {
            "ok": True,
            "version": VERSION,
            "protocol": "kernelyra-api/v1",
        }
        if not has_user_access(request):
            return public
        return {
            **public,
            "pid": os.getpid(),
            "workspace": str(workspace.root),
            "engine": "single-daemon",
            "tensorflow_installed": importlib.util.find_spec("tensorflow") is not None,
            "tensorflow_loaded": "tensorflow" in __import__("sys").modules,
            "resource_enforcement": workspace.hardware.get("resource_enforcement", {}),
            "user_routes": "user_session_required",
            "agent_routes": "scoped_agent_session_required",
        }

    @app.post("/api/v1/mcp/sessions", status_code=201)
    def issue_agent_session(body: AgentSessionRequest, request: Request) -> dict[str, Any]:
        if not has_agent_bootstrap_access(request):
            raise AccessDeniedError("Agent session bootstrap requires agent.secret")
        now = time.time()
        for token, session in list(agent_sessions.items()):
            if session.expires_at <= now:
                agent_sessions.pop(token, None)
        policy = AgentPolicy.load(workspace.root)
        session_id = secrets.token_hex(16)
        session_token = secrets.token_urlsafe(48)
        expires_at = now + body.ttl_seconds
        agent_sessions[session_token] = ScopedAgentSession(
            session_id=session_id,
            client_id=body.client_id,
            expires_at=expires_at,
            policy=policy,
        )
        workspace.storage.log_action(
            "mcp_bootstrap",
            "agent.session.issue",
            {
                "session_id": session_id,
                "client_id": body.client_id,
                "workspace": str(policy.workspace),
                "allowed_roots": [str(path) for path in policy.allowed_roots],
                "allowed_actions": sorted(policy.allowed_actions),
                "expires_at": expires_at,
            },
        )
        return {
            "session_token": session_token,
            "session_id": session_id,
            "client_id": body.client_id,
            "workspace": str(policy.workspace),
            "allowed_roots": [str(path) for path in policy.allowed_roots],
            "allowed_actions": sorted(policy.allowed_actions),
            "expires_at": expires_at,
        }

    @app.get("/api/v1/agent-sessions")
    def list_agent_sessions() -> list[dict[str, Any]]:
        now = time.time()
        return [
            {
                "session_id": session.session_id,
                "client_id": session.client_id,
                "expires_at": session.expires_at,
                "expired": session.expires_at <= now,
                "allowed_roots": [str(path) for path in session.policy.allowed_roots],
                "allowed_actions": sorted(session.policy.allowed_actions),
            }
            for session in agent_sessions.values()
        ]

    @app.delete("/api/v1/agent-sessions/{session_id}")
    def revoke_agent_session(session_id: str) -> dict[str, Any]:
        revoked = False
        for token, session in list(agent_sessions.items()):
            if secrets.compare_digest(session.session_id, session_id):
                agent_sessions.pop(token, None)
                revoked = True
        workspace.storage.log_action("local_user", "agent.session.revoke", {"session_id": session_id, "revoked": revoked})
        return {"session_id": session_id, "revoked": revoked}

    @app.post("/api/v1/approvals/revoke")
    def revoke_approval(body: TokenRequest) -> dict[str, Any]:
        return {"revoked": workspace.storage.revoke_approval(body.token)}

    @app.get("/api/state")
    @app.get("/api/v1/state")
    def state() -> dict[str, Any]:
        return workspace.runtime.snapshot()

    @app.post("/api/batch/plan")
    @app.post("/api/v1/batch/plan")
    def batch_plan(body: BatchPlanRequest) -> dict[str, Any]:
        plan = batch_plan_for(workspace, body.dataset, body.profile, body.batch_mode, body.batch_size, body.ram)
        return {"dataset": body.dataset, **plan.to_dict()}

    def create_run_record(body: RunRequest, *, allow_start: bool) -> dict[str, Any]:
        if body.model_path:
            inspected = workspace.datasets.inspect(body.model_path)
            if inspected.get("kind") != "model" or not inspected.get("fine_tune"):
                raise ValueError("Для этого формата модели не установлен backend дообучения")
        handle = workspace.create_run(
            RunConfig(
                dataset=body.dataset,
                backend=body.backend,
                objective=body.objective,
                architecture=body.architecture,
                model_format=body.model_format,
                name=body.name,
                mode=body.mode,
                profile=body.profile,
                priority=body.priority,
                target_metric=body.target_score,
                batch_mode=body.batch_mode,
                batch_size=body.batch_size,
                accept_batch_risk=body.accept_batch_risk,
                max_steps=body.max_steps,
                cpu=body.cpu,
                ram=body.ram,
                gpu=body.gpu,
                base_run_id=body.base_run_id,
                model_path=body.model_path,
                seed=body.seed,
                learning_rate=body.learning_rate,
                weight_decay=body.weight_decay,
                hidden_layers=body.hidden_layers,
                precision=body.precision,
                data_workers=body.data_workers,
                prefetch=body.prefetch,
                evaluation_interval=body.evaluation_interval,
                min_improvement=body.min_improvement,
                degradation_margin=body.degradation_margin,
                degradation_patience=body.degradation_patience,
                early_stopping_patience=body.early_stopping_patience,
                target_patience=body.target_patience,
            )
        )
        return (handle.start() if allow_start and body.start else handle.info).to_dict()

    def page(items: list[Any], limit: int, offset: int) -> list[Any]:
        safe_limit = max(1, min(500, int(limit)))
        safe_offset = max(0, int(offset))
        return items[safe_offset : safe_offset + safe_limit]

    def run_logs_payload(run_id: str, limit: int = 100) -> list[dict[str, Any]]:
        workspace.runs.get(run_id)
        events = workspace.storage.recent_actions(limit=max(100, min(1000, int(limit) * 5)))
        selected = [event for event in events if event.get("payload", {}).get("run_id") == run_id]
        return selected[: max(1, min(500, int(limit)))]

    def export_run_payload(run_id: str) -> dict[str, Any]:
        run = workspace.runs.get(run_id).info
        dataset = workspace.datasets.get(run.dataset)
        return {
            "contract_version": "kernelyra-run-export/1",
            "run": {
                "id": run.id,
                "name": run.name,
                "dataset": run.dataset,
                "backend": run.backend,
                "effective_backend": run.effective_backend,
                "objective": run.objective,
                "architecture": run.architecture,
                "model_format": run.model_format,
                "profile": run.profile,
                "priority": run.priority,
                "target_score": run.target_score,
                "batch_mode": run.batch_mode,
                "batch_size": run.batch_size,
                "max_steps": run.max_steps,
                "seed": run.seed,
                "learning_rate": run.learning_rate,
                "weight_decay": run.weight_decay,
                "hidden_layers": list(run.hidden_layers),
                "precision": run.precision,
                "data_workers": run.data_workers,
                "prefetch": run.prefetch,
                "base_run_id": run.base_run_id,
                "model_filename": Path(run.model_path).name if run.model_path else None,
                "status": run.status,
                "step": run.step,
                "metrics": run.metrics,
                "best_score": run.best_score,
                "best_step": run.best_step,
                "termination_reason": run.termination_reason,
                "worker_protocol": run.worker_protocol,
                "resource_enforcement": run.resource_enforcement,
                "environment": run.environment_manifest,
                "checkpoint": run.checkpoint,
                "created_at": run.created_at,
            },
            "dataset": {
                "id": dataset.id,
                "source_name": dataset.source,
                "sha256": dataset.sha256,
                "records": dataset.records,
                "features": dataset.features,
                "target": dataset.target,
                "format": dataset.format,
                "task_types": dataset.task_types,
                "schema": dataset.schema,
                "manifest": dataset.manifest,
            },
        }

    @app.post("/api/runs", status_code=201, response_model=RunInfo)
    @app.post("/api/v1/runs", status_code=201, response_model=RunInfo)
    def create_run(body: RunRequest) -> dict[str, Any]:
        return create_run_record(body, allow_start=True)

    @app.post("/api/v1/mcp/runs", status_code=201, response_model=RunInfo)
    def mcp_create_run(body: RunRequest, request: Request) -> dict[str, Any]:
        require_agent_action(request, "run.create")
        if body.model_path:
            raise ApprovalError("Agent не может передавать model_path через create_run")
        return create_run_record(body, allow_start=False)

    @app.get("/api/v1/mcp/runs", response_model=list[RunInfo])
    def mcp_list_runs(request: Request, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        require_agent_action(request, "run.list")
        return [run.to_dict() for run in page(workspace.runs.list(), limit, offset)]

    @app.get("/api/v1/mcp/runs/{run_id}", response_model=RunInfo)
    def mcp_get_run(run_id: str, request: Request) -> dict[str, Any]:
        require_agent_action(request, "run.read")
        return workspace.runs.get(run_id).info.to_dict()

    @app.get("/api/v1/runs", response_model=list[RunInfo])
    def list_runs(limit: int = 100, offset: int = 0, status: str | None = None) -> list[dict[str, Any]]:
        runs = workspace.runs.list()
        if status:
            runs = [run for run in runs if run.status == status]
        return [run.to_dict() for run in page(runs, limit, offset)]

    @app.get("/api/v1/runs/{run_id}", response_model=RunInfo)
    def get_run(run_id: str) -> dict[str, Any]:
        return workspace.runs.get(run_id).info.to_dict()

    @app.get("/api/v1/runs/{run_id}/metrics", response_model=RunMetricsResponse)
    def get_run_metrics(run_id: str) -> dict[str, Any]:
        run = workspace.runs.get(run_id).info
        return {"run_id": run.id, "step": run.step, "metrics": run.metrics}

    @app.get("/api/v1/mcp/runs/{run_id}/metrics", response_model=RunMetricsResponse)
    def mcp_get_run_metrics(run_id: str, request: Request) -> dict[str, Any]:
        require_agent_action(request, "run.metrics")
        return get_run_metrics(run_id)

    @app.get("/api/v1/runs/{run_id}/logs")
    def get_run_logs(run_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return run_logs_payload(run_id, limit)

    @app.get("/api/v1/mcp/runs/{run_id}/logs")
    def mcp_get_run_logs(run_id: str, request: Request, limit: int = 100) -> list[dict[str, Any]]:
        require_agent_action(request, "logs.read")
        return run_logs_payload(run_id, limit)

    @app.get("/api/v1/runs/{run_id}/export")
    def export_run(run_id: str) -> dict[str, Any]:
        return export_run_payload(run_id)

    @app.post("/api/v1/mcp/runs/{run_id}/export")
    def mcp_export_run(run_id: str, body: MCPExportRequest, request: Request) -> dict[str, Any]:
        require_agent_action(request, "run.export")
        workspace.runs.get(run_id)
        if not workspace.storage.consume_approval(body.approval_token, "run.export", run_id):
            raise ApprovalError("Нужен действующий одноразовый approval-токен пользователя")
        result = export_run_payload(run_id)
        workspace.storage.log_action("mcp", "run.export", {"run_id": run_id})
        return result

    @app.post("/api/runs/{run_id}/command")
    @app.post("/api/v1/runs/{run_id}/command")
    def run_command(run_id: str, body: CommandRequest) -> dict[str, Any]:
        run = workspace.runtime.command(run_id, body.command, actor="api")
        expected = {"pause": "paused", "stop": "stopped"}.get(body.command)
        if expected is not None and run.status != expected:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                current = workspace.storage.get_run(run_id)
                if current is None or current.status == expected:
                    run = current or run
                    break
                if current.status not in {"pausing", "stopping"}:
                    run = current
                    break
                time.sleep(.01)
        return run.to_dict()

    @app.post("/api/v1/mcp/runs/{run_id}/command")
    def mcp_run_command(run_id: str, body: MCPCommandRequest, request: Request) -> dict[str, Any]:
        require_agent_action(request, f"run.{body.command}")
        if body.command in {"start", "resume"}:
            if not body.approval_token or not workspace.storage.consume_approval(
                body.approval_token, f"run.{body.command}", run_id
            ):
                raise ApprovalError("Нужен действующий одноразовый approval-токен пользователя")
        return workspace.runtime.command(run_id, body.command, actor="mcp").to_dict()

    @app.post("/api/v1/approvals", status_code=201)
    def issue_approval(body: ApprovalRequest, request: Request) -> dict[str, Any]:
        if not has_user_access(request):
            raise ApprovalError("Выпуск approval разрешён только локальному CLI с daemon-secret")
        if body.action not in {"run.start", "run.resume", "run.export", "dataset.import"}:
            raise ValueError("Неподдерживаемое approval-действие")
        resource_id = body.resource_id
        if body.action.startswith("run."):
            workspace.runs.get(resource_id)
        else:
            resource_id = str(Path(resource_id).expanduser().resolve())
        token, expires_at = workspace.storage.issue_approval(
            body.action, resource_id, body.ttl_seconds, actor="user_cli"
        )
        return {
            "approval_token": token,
            "action": body.action,
            "resource_id": resource_id,
            "expires_at": expires_at,
            "one_time": True,
        }

    @app.post("/api/rebalance")
    @app.post("/api/v1/rebalance")
    def rebalance() -> dict[str, Any]:
        return workspace.runtime.rebalance()

    @app.post("/api/paths/inspect")
    @app.post("/api/v1/paths/inspect")
    def inspect_path(body: PathRequest) -> dict[str, Any]:
        return workspace.datasets.inspect(body.path)

    @app.post("/api/v1/mcp/paths/inspect")
    def mcp_inspect_path(body: PathRequest, request: Request) -> dict[str, Any]:
        session = require_agent_action(request, "path.inspect")
        approved = session.policy.require_path(body.path)
        return workspace.datasets.inspect(approved)

    def import_path(path: str, target: str | None) -> dict[str, Any]:
        inspected = workspace.datasets.inspect(path)
        source = inspected.get("candidate_path") or path
        if not inspected.get("trainable"):
            raise ValueError("Формат распознан, но обучающий ingestor для него не установлен")
        dataset = workspace.datasets.import_file(source, target)
        return {
            **dataset.to_dict(),
            "reused": False,
            "native_probe": {"engine": inspected.get("engine", "format-router")},
        }

    @app.post("/api/datasets/from-path", status_code=201, response_model=DatasetInfo)
    @app.post("/api/v1/datasets/from-path", status_code=201, response_model=DatasetInfo)
    def dataset_from_path(body: PathRequest) -> dict[str, Any]:
        return import_path(body.path, body.target)

    @app.post("/api/v1/mcp/datasets/from-path", status_code=201, response_model=DatasetInfo)
    def mcp_dataset_from_path(body: MCPDatasetRequest, request: Request) -> dict[str, Any]:
        session = require_agent_action(request, "dataset.import")
        resource_id = str(session.policy.require_path(body.path))
        if not workspace.storage.consume_approval(body.approval_token, "dataset.import", resource_id):
            raise ApprovalError("Нужен действующий одноразовый approval-токен пользователя")
        result = import_path(resource_id, body.target)
        workspace.storage.log_action("mcp", "dataset.import", {"path": resource_id, "dataset_id": result["id"]})
        return result

    @app.get("/api/v1/datasets", response_model=list[DatasetInfo])
    def list_datasets(limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return [item.to_dict() for item in page(workspace.datasets.list(), limit, offset)]

    @app.get("/api/v1/mcp/datasets", response_model=list[DatasetInfo])
    def mcp_list_datasets(request: Request, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        require_agent_action(request, "dataset.list")
        return [item.to_dict() for item in page(workspace.datasets.list(), limit, offset)]

    @app.get("/api/v1/datasets/{dataset_id}", response_model=DatasetInfo)
    def get_dataset(dataset_id: str) -> dict[str, Any]:
        return workspace.datasets.get(dataset_id).to_dict()

    @app.get("/api/v1/mcp/datasets/{dataset_id}", response_model=DatasetInfo)
    def mcp_get_dataset(dataset_id: str, request: Request) -> dict[str, Any]:
        require_agent_action(request, "dataset.read")
        return get_dataset(dataset_id)

    @app.delete("/api/v1/datasets/{dataset_id}", status_code=204)
    def remove_dataset(dataset_id: str) -> Response:
        workspace.datasets.remove(dataset_id)
        workspace.storage.log_action("api", "dataset.remove", {"dataset_id": dataset_id})
        return Response(status_code=204)

    @app.post("/api/datasets", status_code=201)
    @app.post("/api/v1/datasets", status_code=201)
    async def upload_dataset(file: UploadFile = File(...), target: str | None = None) -> dict[str, Any]:  # noqa: B008
        contents = await file.read(50 * 1024 * 1024 + 1)
        if len(contents) > 50 * 1024 * 1024:
            raise PayloadTooLargeError("Файл больше безопасного upload-лимита 50 MB")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(file.filename or "dataset.csv").name)
        upload_dir = workspace.state_dir / "uploads"
        upload_dir.mkdir(exist_ok=True)
        pending = upload_dir / f"pending-{__import__('uuid').uuid4().hex[:8]}-{safe_name}"
        pending.write_bytes(contents)
        try:
            inspected = workspace.datasets.inspect(pending)
            dataset = workspace.datasets.import_file(pending, target, source_name=file.filename or safe_name)
            return {**dataset.to_dict(), "native_probe": {"engine": inspected.get("engine", "format-router")}}
        finally:
            pending.unlink(missing_ok=True)

    @app.get("/api/model-formats")
    @app.get("/api/v1/model-formats")
    def model_formats() -> dict[str, Any]:
        return {
            "models": list(CHECKPOINT_FORMATS),
            "formats": describe_formats(),
            "ingestors": workspace.datasets.ingestors.describe(),
            "recognized_routes": workspace.datasets.router.route_count,
        }

    @app.get("/api/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        return workspace.capabilities

    @app.get("/api/v1/mcp/capabilities")
    def mcp_capabilities(request: Request) -> dict[str, Any]:
        require_agent_action(request, "capabilities.read")
        return workspace.capabilities

    @app.get("/api/v1/hardware")
    def hardware() -> dict[str, Any]:
        return workspace.hardware

    @app.get("/api/v1/mcp/hardware")
    def mcp_hardware(request: Request) -> dict[str, Any]:
        require_agent_action(request, "hardware.read")
        return workspace.hardware

    @app.get("/api/v1/logs")
    def logs(limit: int = 50) -> list[dict[str, Any]]:
        return workspace.storage.recent_actions(limit)

    @app.get("/api/v1/events")
    async def events(request: Request, after: int = 0) -> StreamingResponse:
        header = request.headers.get("Last-Event-ID", "")
        try:
            last_id = max(int(after), int(header) if header else 0)
        except ValueError:
            last_id = max(0, int(after))

        async def stream() -> AsyncIterator[str]:
            nonlocal last_id
            while not await request.is_disconnected():
                pending = workspace.storage.events_since(last_id, 100)
                if not pending:
                    yield ": keepalive\n\n"
                    await asyncio.sleep(.5)
                    continue
                for item in pending:
                    last_id = int(item["id"])
                    payload = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                    yield f"id: {last_id}\nevent: kernelyra\ndata: {payload}\n\n"
                await asyncio.sleep(0)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/v1/mcp/logs")
    def mcp_logs(request: Request, limit: int = 50) -> list[dict[str, Any]]:
        require_agent_action(request, "logs.read")
        return workspace.storage.recent_actions(limit)

    @app.get("/")
    def headless_root() -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": "Kernelyra is terminal-first and has no web UI", "cli": "kernelyra --help"},
        )

    return app


def serve(
    workspace_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    assets_root: str | Path | None = None,
    *,
    acquire_lock: bool = True,
    api_token: str | None = None,
) -> None:
    import uvicorn

    validate_daemon_bind(host, api_token)
    enable_ui = host.strip().lower() in LOOPBACK_HOSTS
    if not enable_ui:
        print(
            f"WARNING: Kernelyra доступен по сети на {host}:{port}; API защищён bearer token, UI отключён, TLS должен обеспечить reverse proxy.",
            file=sys.stderr,
            flush=True,
        )

    if not acquire_lock:
        uvicorn.run(create_app(workspace_path, assets_root, api_token, enable_ui=enable_ui, allowed_hosts=[host]), host=host, port=port, log_level="info")
        return
    from .daemon import DaemonLock

    with DaemonLock(workspace_path, host, port):
        uvicorn.run(create_app(workspace_path, assets_root, api_token, enable_ui=enable_ui, allowed_hosts=[host]), host=host, port=port, log_level="info")
