from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import asdict
from typing import Any
from urllib import error, parse, request

from .errors import DaemonUnavailableError, KernelyraError
from .models import RunConfig


class RemoteError(KernelyraError):
    def __init__(self, message: str, status: int, error_type: str = "RemoteError"):
        self.status = status
        self.error_type = error_type
        super().__init__(message)


class DaemonClient:
    """Small dependency-free client for the single Kernelyra control plane."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        timeout: float = 10.0,
        api_token: str | None = None,
        user_secret: str | None = None,
        agent_secret: str | None = None,
        agent_client_id: str | None = None,
        retries: int = 2,
    ):
        normalized_url = base_url.rstrip("/")
        parsed_url = parse.urlsplit(normalized_url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.hostname
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.path not in {"", "/"}
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ValueError("base_url must be an absolute HTTP(S) origin without credentials or a path")
        self.base_url = normalized_url
        self.timeout = timeout
        self.api_token = api_token
        self.user_secret = user_secret
        self.agent_secret = agent_secret
        self.agent_client_id = agent_client_id or f"kernelyra-{os.getpid()}-{secrets.token_hex(8)}"
        self.agent_session: str | None = None
        self.agent_session_expires_at = 0.0
        self.retries = max(0, min(5, int(retries)))

    def request(
        self,
        method: str,
        path: str,
        body: Any | None = None,
        extra_headers: dict[str, str] | None = None,
        _retry_agent: bool = True,
        _network_attempt: int = 0,
    ) -> Any:
        is_agent_route = path.startswith("/api/v1/mcp/") and path != "/api/v1/mcp/sessions"
        if (
            is_agent_route
            and self.agent_secret
            and (not self.agent_session or self.agent_session_expires_at <= time.time() + 30)
        ):
            self.create_agent_session()
        payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        if self.user_secret:
            headers["X-Kernelyra-User-Secret"] = self.user_secret
        if self.agent_secret and path == "/api/v1/mcp/sessions":
            headers["X-Kernelyra-Agent-Secret"] = self.agent_secret
        if self.agent_session and is_agent_route:
            headers["X-Kernelyra-Agent-Session"] = self.agent_session
            headers["X-Kernelyra-Agent-Client"] = self.agent_client_id
        if extra_headers:
            headers.update(extra_headers)
        if payload is not None:
            headers["Content-Type"] = "application/json"
        # ``base_url`` is validated as an HTTP(S) origin above and every API path is appended to it.
        call = request.Request(self.base_url + path, data=payload, headers=headers, method=method)
        try:
            with request.urlopen(call, timeout=self.timeout) as response:  # nosec B310
                raw = response.read()
        except error.HTTPError as exc:
            raw = exc.read()
            try:
                detail = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                detail = {"error": str(exc), "type": "HTTPError"}
            error_type = str(detail.get("type") or "HTTPError")
            if error_type == "AgentSessionRequired" and self.agent_secret and _retry_agent:
                self.agent_session = None
                self.agent_session_expires_at = 0.0
                self.create_agent_session()
                return self.request(
                    method,
                    path,
                    body,
                    extra_headers,
                    _retry_agent=False,
                    _network_attempt=_network_attempt,
                )
            raise RemoteError(
                str(detail.get("error") or exc.reason),
                exc.code,
                error_type,
            ) from None
        except (error.URLError, TimeoutError, OSError) as exc:
            if method in {"GET", "HEAD"} and _network_attempt < self.retries:
                time.sleep(min(.5, .05 * (2**_network_attempt)))
                return self.request(
                    method,
                    path,
                    body,
                    extra_headers,
                    _retry_agent=_retry_agent,
                    _network_attempt=_network_attempt + 1,
                )
            raise DaemonUnavailableError(
                f"Kernelyra daemon недоступен по адресу {self.base_url}. "
                "Запустите: kernelyra daemon start"
            ) from exc
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))

    def _dict_request(
        self,
        method: str,
        path: str,
        body: Any | None = None,
        extra_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = self.request(method, path, body, extra_headers, **kwargs)
        if not isinstance(result, dict):
            raise RemoteError("Daemon returned a malformed object response", 502, "ProtocolError")
        return result

    def _list_request(
        self,
        method: str,
        path: str,
        body: Any | None = None,
        extra_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        result = self.request(method, path, body, extra_headers, **kwargs)
        if not isinstance(result, list) or any(not isinstance(item, dict) for item in result):
            raise RemoteError("Daemon returned a malformed list response", 502, "ProtocolError")
        return result

    def health(self) -> dict[str, Any]:
        return self._dict_request("GET", "/api/v1/health")

    def capabilities(self) -> dict[str, Any]:
        return self._dict_request("GET", "/api/v1/capabilities")

    def create_agent_session(self, ttl_seconds: int = 300) -> dict[str, Any]:
        if not self.agent_secret:
            raise KernelyraError("agent.secret is required to create an agent session")
        result = self._dict_request(
            "POST",
            "/api/v1/mcp/sessions",
            {"client_id": self.agent_client_id, "ttl_seconds": ttl_seconds},
            _retry_agent=False,
        )
        self.agent_session = str(result["session_token"])
        self.agent_session_expires_at = float(result["expires_at"])
        return result

    def state(self) -> dict[str, Any]:
        return self._dict_request("GET", "/api/v1/state")

    def inspect(self, path: str) -> dict[str, Any]:
        return self._dict_request("POST", "/api/v1/paths/inspect", {"path": path})

    def add_dataset(self, path: str, target: str | None = None) -> dict[str, Any]:
        return self._dict_request("POST", "/api/v1/datasets/from-path", {"path": path, "target": target})

    def list_datasets(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        query = parse.urlencode({"limit": limit, "offset": offset})
        return self._list_request("GET", f"/api/v1/datasets?{query}")

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self._dict_request("GET", f"/api/v1/datasets/{parse.quote(dataset_id, safe='')}")

    def remove_dataset(self, dataset_id: str) -> None:
        self.request("DELETE", f"/api/v1/datasets/{parse.quote(dataset_id, safe='')}")

    def create_run(self, config: RunConfig) -> dict[str, Any]:
        body = asdict(config)
        body["target_score"] = body.pop("target_metric")
        body["start"] = False
        return self._dict_request("POST", "/api/v1/runs", body)

    def mcp_create_run(self, config: RunConfig) -> dict[str, Any]:
        body = asdict(config)
        body["target_score"] = body.pop("target_metric")
        body["start"] = False
        return self._dict_request("POST", "/api/v1/mcp/runs", body)

    def mcp_inspect(self, path: str) -> dict[str, Any]:
        return self._dict_request("POST", "/api/v1/mcp/paths/inspect", {"path": path})

    def mcp_import_dataset(self, path: str, target: str | None, approval_token: str) -> dict[str, Any]:
        return self._dict_request(
            "POST",
            "/api/v1/mcp/datasets/from-path",
            {"path": path, "target": target, "approval_token": approval_token},
        )

    def mcp_capabilities(self) -> dict[str, Any]:
        return self._dict_request("GET", "/api/v1/mcp/capabilities")

    def mcp_list_datasets(self) -> list[dict[str, Any]]:
        return self._list_request("GET", "/api/v1/mcp/datasets")

    def mcp_get_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self._dict_request("GET", f"/api/v1/mcp/datasets/{parse.quote(dataset_id, safe='')}")

    def mcp_list_runs(self) -> list[dict[str, Any]]:
        return self._list_request("GET", "/api/v1/mcp/runs")

    def mcp_get_run(self, run_id: str) -> dict[str, Any]:
        return self._dict_request("GET", f"/api/v1/mcp/runs/{parse.quote(run_id, safe='')}")

    def mcp_hardware(self) -> dict[str, Any]:
        return self._dict_request("GET", "/api/v1/mcp/hardware")

    def mcp_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._list_request("GET", f"/api/v1/mcp/logs?limit={max(1, min(1000, limit))}")

    def list_runs(self, limit: int = 100, offset: int = 0, status: str | None = None) -> list[dict[str, Any]]:
        values: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            values["status"] = status
        return self._list_request("GET", f"/api/v1/runs?{parse.urlencode(values)}")

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._dict_request("GET", f"/api/v1/runs/{parse.quote(run_id, safe='')}")

    def get_run_metrics(self, run_id: str) -> dict[str, Any]:
        return self._dict_request("GET", f"/api/v1/runs/{parse.quote(run_id, safe='')}/metrics")

    def get_run_logs(self, run_id: str, limit: int = 100) -> list[dict[str, Any]]:
        encoded = parse.quote(run_id, safe="")
        return self._list_request("GET", f"/api/v1/runs/{encoded}/logs?limit={max(1, min(500, limit))}")

    def export_run(self, run_id: str) -> dict[str, Any]:
        return self._dict_request("GET", f"/api/v1/runs/{parse.quote(run_id, safe='')}/export")

    def mcp_get_run_metrics(self, run_id: str) -> dict[str, Any]:
        return self._dict_request("GET", f"/api/v1/mcp/runs/{parse.quote(run_id, safe='')}/metrics")

    def mcp_get_run_logs(self, run_id: str, limit: int = 100) -> list[dict[str, Any]]:
        encoded = parse.quote(run_id, safe="")
        return self._list_request("GET", f"/api/v1/mcp/runs/{encoded}/logs?limit={max(1, min(500, limit))}")

    def mcp_export_run(self, run_id: str, approval_token: str) -> dict[str, Any]:
        encoded = parse.quote(run_id, safe="")
        return self._dict_request(
            "POST", f"/api/v1/mcp/runs/{encoded}/export", {"approval_token": approval_token}
        )

    def command(self, run_id: str, command: str) -> dict[str, Any]:
        encoded = parse.quote(run_id, safe="")
        return self._dict_request("POST", f"/api/v1/runs/{encoded}/command", {"command": command})

    def mcp_command(self, run_id: str, command: str, approval_token: str) -> dict[str, Any]:
        encoded = parse.quote(run_id, safe="")
        return self._dict_request(
            "POST",
            f"/api/v1/mcp/runs/{encoded}/command",
            {"command": command, "approval_token": approval_token},
        )

    def issue_approval(
        self,
        action: str,
        resource_id: str,
        ttl_seconds: int = 300,
        user_secret: str | None = None,
    ) -> dict[str, Any]:
        return self._dict_request(
            "POST",
            "/api/v1/approvals",
            {"action": action, "resource_id": resource_id, "ttl_seconds": ttl_seconds},
            {"X-Kernelyra-User-Secret": user_secret} if user_secret else None,
        )

    def revoke_approval(self, token: str) -> dict[str, Any]:
        return self._dict_request("POST", "/api/v1/approvals/revoke", {"token": token})

    def logs(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._list_request("GET", f"/api/v1/logs?limit={max(1, min(1000, limit))}")

    def close(self) -> None:
        self.agent_session = None
        self.agent_session_expires_at = 0.0
        self.user_secret = None
        self.agent_secret = None

    def __enter__(self) -> DaemonClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
