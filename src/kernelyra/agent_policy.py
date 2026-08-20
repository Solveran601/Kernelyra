from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import AccessDeniedError, ConfigurationError

DEFAULT_AGENT_ACTIONS = frozenset(
    {
        "dataset.import",
        "dataset.list",
        "dataset.read",
        "capabilities.read",
        "hardware.read",
        "logs.read",
        "path.inspect",
        "run.create",
        "run.list",
        "run.metrics",
        "run.export",
        "run.pause",
        "run.read",
        "run.resume",
        "run.start",
        "run.stop",
    }
)


@dataclass(frozen=True, slots=True)
class AgentPolicy:
    workspace: Path
    allowed_roots: tuple[Path, ...]
    allowed_actions: frozenset[str]

    @classmethod
    def load(cls, workspace: str | Path) -> AgentPolicy:
        root = Path(workspace).expanduser().resolve()
        config = root / "kernelyra.toml"
        raw: dict[str, Any] = {}
        if config.exists():
            with config.open("rb") as handle:
                raw = tomllib.load(handle).get("mcp", {}).get("permissions", {})

        env_roots = [part for part in os.environ.get("KERNELYRA_ALLOWED_ROOTS", "").split(os.pathsep) if part]
        configured_roots = raw["allowed_roots"] if "allowed_roots" in raw else (env_roots or [str(root)])
        if not isinstance(configured_roots, list) or not all(
            isinstance(raw_root, str) for raw_root in configured_roots
        ):
            raise ConfigurationError("mcp.permissions.allowed_roots должен быть списком строк")
        resolved_roots: list[Path] = []
        for raw_root in configured_roots:
            candidate = Path(str(raw_root)).expanduser()
            if not candidate.is_absolute():
                candidate = config.parent / candidate
            resolved_roots.append(candidate.resolve())

        configured_actions = (
            raw["allowed_actions"] if "allowed_actions" in raw else sorted(DEFAULT_AGENT_ACTIONS)
        )
        if not isinstance(configured_actions, list) or not all(
            isinstance(action, str) for action in configured_actions
        ):
            raise ConfigurationError("mcp.permissions.allowed_actions должен быть списком строк")
        actions = frozenset(configured_actions)
        unknown = actions - DEFAULT_AGENT_ACTIONS
        if unknown:
            raise ConfigurationError(
                "Неизвестные agent actions: " + ", ".join(sorted(unknown))
            )
        return cls(root, tuple(resolved_roots), actions)

    def require_action(self, action: str) -> None:
        if action not in self.allowed_actions:
            raise AccessDeniedError(f"Agent session does not allow action: {action}")

    def require_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.workspace / path
        path = path.resolve()
        if not any(path == root or root in path.parents for root in self.allowed_roots):
            raise AccessDeniedError("Access denied: path is outside server-approved roots")
        return path
