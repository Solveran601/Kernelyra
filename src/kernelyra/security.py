from __future__ import annotations

import os
import secrets
from pathlib import Path

from .errors import ConfigurationError

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def validate_daemon_bind(host: str, api_token: str | None) -> None:
    if host.strip().lower() in LOOPBACK_HOSTS:
        return
    if not api_token:
        raise ConfigurationError(
            f"Отказ небезопасного binding на {host}: для не-loopback адреса обязателен --api-token "
            "или KERNELYRA_API_TOKEN. Используйте 127.0.0.1 для локального режима."
        )
    if len(api_token) < 24:
        raise ConfigurationError("API token для сетевого режима должен содержать не менее 24 символов")


def _ensure_secret(state_dir: Path, filename: str) -> str:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / filename
    try:
        return path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        pass
    value = secrets.token_urlsafe(48)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_text(encoding="ascii").strip()
    with os.fdopen(descriptor, "w", encoding="ascii", newline="") as handle:
        handle.write(value)
    if os.name != "nt":
        os.chmod(path, 0o600)
    return value


def ensure_user_secret(state_dir: Path) -> str:
    """Create/read the credential for trusted local user clients."""
    return _ensure_secret(state_dir, "daemon.secret")


def ensure_agent_secret(state_dir: Path) -> str:
    """Create/read the separate least-privilege MCP/agent credential."""
    return _ensure_secret(state_dir, "agent.secret")


def load_user_secret(workspace: str | Path) -> str:
    path = Path(workspace).expanduser().resolve() / ".kernelyra" / "daemon.secret"
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise ConfigurationError(
            "Локальный daemon-secret не найден. Сначала запустите: kernelyra daemon start"
        ) from exc
    if not value:
        raise ConfigurationError("Локальный daemon-secret пуст или повреждён")
    return value


def load_agent_secret(workspace: str | Path) -> str:
    path = Path(workspace).expanduser().resolve() / ".kernelyra" / "agent.secret"
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise ConfigurationError(
            "Локальный agent-secret не найден. Сначала запустите: kernelyra daemon start"
        ) from exc
    if not value:
        raise ConfigurationError("Локальный agent-secret пуст или повреждён")
    return value
