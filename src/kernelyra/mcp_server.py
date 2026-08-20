from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import DaemonClient
from .errors import AccessDeniedError, ConfigurationError
from .models import RunConfig
from .security import load_agent_secret


@dataclass(frozen=True, slots=True)
class MCPPermissions:
    allowed_roots: tuple[Path, ...]
    path_base: Path
    allow_read: bool = True
    allow_write: bool = False
    allow_delete: bool = False

    @classmethod
    def load(cls, workspace: Path, config_path: str | Path | None = None) -> MCPPermissions:
        config = Path(config_path).resolve() if config_path else workspace / "kernelyra.toml"
        relative_base = config.parent
        raw: dict[str, Any] = {}
        if config.exists():
            with config.open("rb") as handle:
                raw = tomllib.load(handle).get("mcp", {}).get("permissions", {})
        env_roots = [part for part in os.environ.get("KERNELYRA_ALLOWED_ROOTS", "").split(os.pathsep) if part]
        roots = raw.get("allowed_roots") or env_roots or [str(workspace)]
        resolved_roots = []
        for root in roots:
            candidate = Path(root).expanduser()
            if not candidate.is_absolute():
                candidate = relative_base / candidate
            resolved_roots.append(candidate.resolve())
        return cls(
            tuple(resolved_roots),
            relative_base.resolve(),
            bool(raw.get("allow_read", True)),
            bool(raw.get("allow_write", False)),
            bool(raw.get("allow_delete", False)),
        )

    def require_path(self, raw_path: str | Path, write: bool = False) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.path_base / path
        path = path.resolve()
        if write and not self.allow_write:
            raise AccessDeniedError("Access denied: MCP write access is disabled")
        if not write and not self.allow_read:
            raise AccessDeniedError("Access denied: MCP read access is disabled")
        if not any(path == root or root in path.parents for root in self.allowed_roots):
            raise AccessDeniedError("Access denied: path is outside approved workspace roots")
        return path


def build_mcp(
    workspace_path: str | Path,
    daemon_url: str = "http://127.0.0.1:8765",
    config_path: str | Path | None = None,
    api_token: str | None = None,
) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as error:
        if error.name != "mcp" and not str(error.name).startswith("mcp."):
            raise
        raise ConfigurationError(
            'MCP support is not installed. Run: pip install "kernelyra-ai[mcp]"'
        ) from None

    root = Path(workspace_path).resolve()
    permissions = MCPPermissions.load(root, config_path)
    client = DaemonClient(
        daemon_url,
        api_token=api_token,
        agent_secret=load_agent_secret(root),
    )
    client.health()
    client.create_agent_session()
    mcp = FastMCP(
        "Kernelyra",
        instructions=(
            "Manage Kernelyra through its single daemon. Starting or resuming training and importing "
            "a dataset requires a short-lived one-time approval token issued separately by the user."
        ),
        json_response=True,
    )

    @mcp.tool(name="kernelyra.inspect_path")
    def inspect_path(path: str) -> dict[str, Any]:
        """Safely inspect a path inside approved roots without importing it."""
        approved = permissions.require_path(path)
        return client.mcp_inspect(str(approved))

    @mcp.tool(name="kernelyra.list_datasets")
    def list_datasets() -> list[dict[str, Any]]:
        return client.mcp_list_datasets()

    @mcp.tool(name="kernelyra.inspect_dataset")
    def inspect_dataset(dataset_id: str) -> dict[str, Any]:
        """Return bounded schema and metadata; dataset rows are never returned."""
        return client.mcp_get_dataset(dataset_id)

    @mcp.tool(name="kernelyra.import_dataset")
    def import_dataset(path: str, approval_token: str, target: str | None = None) -> dict[str, Any]:
        """Import using a one-time user-issued ``dataset.import`` approval token."""
        approved = permissions.require_path(path)
        return client.mcp_import_dataset(str(approved), target, approval_token)

    @mcp.tool(name="kernelyra.create_run")
    def create_run(
        dataset: str,
        backend: str = "numpy",
        task: str = "binary_classification",
        architecture: str = "auto",
        model_format: str = "auto",
        target_metric: float = .92,
        batch_mode: str = "auto",
        batch_size: int | None = None,
        name: str = "agent-run",
        evaluation_interval: int | None = None,
        min_improvement: float = .0005,
        degradation_margin: float | None = None,
        degradation_patience: int = 3,
        early_stopping_patience: int = 18,
        target_patience: int = 3,
    ) -> dict[str, Any]:
        """Create a draft run. This cannot start training."""
        return client.mcp_create_run(
            RunConfig(
                dataset=dataset,
                backend=backend,
                objective=task,
                architecture=architecture,
                model_format=model_format,
                target_metric=target_metric,
                batch_mode=batch_mode,
                batch_size=batch_size,
                name=name,
                evaluation_interval=evaluation_interval,
                min_improvement=min_improvement,
                degradation_margin=degradation_margin,
                degradation_patience=degradation_patience,
                early_stopping_patience=early_stopping_patience,
                target_patience=target_patience,
            )
        )

    @mcp.tool(name="kernelyra.start_run")
    def start_run(run_id: str, approval_token: str) -> dict[str, Any]:
        """Start using a one-time user-issued ``run.start`` approval token."""
        return client.mcp_command(run_id, "start", approval_token)

    @mcp.tool(name="kernelyra.pause_run")
    def pause_run(run_id: str) -> dict[str, Any]:
        return client.mcp_command(run_id, "pause", "")

    @mcp.tool(name="kernelyra.resume_run")
    def resume_run(run_id: str, approval_token: str) -> dict[str, Any]:
        """Resume using a one-time user-issued ``run.resume`` approval token."""
        return client.mcp_command(run_id, "resume", approval_token)

    @mcp.tool(name="kernelyra.stop_run")
    def stop_run(run_id: str) -> dict[str, Any]:
        return client.mcp_command(run_id, "stop", "")

    @mcp.tool(name="kernelyra.get_run")
    def get_run(run_id: str) -> dict[str, Any]:
        return client.mcp_get_run(run_id)

    @mcp.tool(name="kernelyra.list_runs")
    def list_runs() -> list[dict[str, Any]]:
        return client.mcp_list_runs()

    @mcp.tool(name="kernelyra.get_hardware")
    def get_hardware() -> dict[str, Any]:
        return client.mcp_hardware()

    @mcp.tool(name="kernelyra.get_logs")
    def get_logs(run_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return client.mcp_get_run_logs(run_id, limit) if run_id else client.mcp_logs(limit)

    @mcp.tool(name="kernelyra.get_metrics")
    def get_metrics(run_id: str) -> dict[str, Any]:
        return client.mcp_get_run_metrics(run_id)

    @mcp.tool(name="kernelyra.get_capabilities")
    def get_capabilities() -> dict[str, Any]:
        return client.mcp_capabilities()

    @mcp.tool(name="kernelyra.list_backends")
    def list_backends() -> list[dict[str, Any]]:
        return list(get_capabilities()["backends"])

    @mcp.tool(name="kernelyra.list_ingestors")
    def list_ingestors() -> list[dict[str, Any]]:
        return list(get_capabilities()["ingestors"])

    @mcp.tool(name="kernelyra.export_run_manifest")
    def export_run_manifest(run_id: str, approval_token: str) -> dict[str, Any]:
        """Export a secret-free manifest with a one-time ``run.export`` approval."""
        return client.mcp_export_run(run_id, approval_token)

    @mcp.resource("kernelyra://system/health")
    def health_resource() -> str:
        return json.dumps(client.health(), ensure_ascii=False)

    @mcp.resource("kernelyra://system/capabilities")
    def capabilities_resource() -> str:
        return json.dumps(get_capabilities(), ensure_ascii=False)

    @mcp.resource("kernelyra://system/hardware")
    def hardware_resource() -> str:
        return json.dumps(client.mcp_hardware(), ensure_ascii=False)

    @mcp.resource("kernelyra://datasets")
    def datasets_resource() -> str:
        return json.dumps(client.mcp_list_datasets(), ensure_ascii=False)

    @mcp.resource("kernelyra://runs")
    def runs_resource() -> str:
        return json.dumps(client.mcp_list_runs(), ensure_ascii=False)

    @mcp.resource("kernelyra://datasets/{dataset_id}")
    def dataset_resource(dataset_id: str) -> str:
        return json.dumps(client.mcp_get_dataset(dataset_id), ensure_ascii=False)

    @mcp.resource("kernelyra://runs/{run_id}")
    def run_resource(run_id: str) -> str:
        return json.dumps(client.mcp_get_run(run_id), ensure_ascii=False)

    @mcp.resource("kernelyra://runs/{run_id}/metrics")
    def run_metrics_resource(run_id: str) -> str:
        return json.dumps(client.mcp_get_run_metrics(run_id), ensure_ascii=False)

    @mcp.resource("kernelyra://runs/{run_id}/logs")
    def run_logs_resource(run_id: str) -> str:
        return json.dumps(client.mcp_get_run_logs(run_id, 100), ensure_ascii=False)

    return mcp


def run_mcp(
    workspace_path: str | Path,
    daemon_url: str = "http://127.0.0.1:8765",
    config_path: str | Path | None = None,
    api_token: str | None = None,
) -> None:
    build_mcp(workspace_path, daemon_url, config_path, api_token).run(transport="stdio")
