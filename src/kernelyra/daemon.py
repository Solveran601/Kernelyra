from __future__ import annotations

import json
import os
import signal
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Any

from .client import DaemonClient
from .errors import ConfigurationError, DaemonUnavailableError
from .security import ensure_agent_secret, ensure_user_secret, load_user_secret, validate_daemon_bind

# Daemon subprocesses use the current interpreter and a fixed module without a shell.


def background_process_options(platform_name: str | None = None) -> dict[str, Any]:
    """Return explicit terminal-detachment options for the current platform."""
    platform = platform_name or os.name
    if platform == "nt":
        new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        detached = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        return {"creationflags": new_group | detached}
    return {"start_new_session": True}


def daemon_url(host: str, port: int) -> str:
    clean = host.strip()
    if clean.startswith("[") and clean.endswith("]"):
        display = clean
    else:
        display = f"[{clean}]" if ":" in clean else clean
    return f"http://{display}:{port}"


def daemon_process_spec(
    root: Path,
    host: str,
    port: int,
    url: str,
    api_token: str | None,
) -> tuple[list[str], dict[str, str]]:
    command = [
        sys.executable,
        "-m",
        "kernelyra.cli",
        "--workspace",
        str(root),
        "--daemon-url",
        url,
        "daemon",
        "foreground",
        "--host",
        host,
        "--port",
        str(port),
    ]
    environment = os.environ.copy()
    if api_token:
        environment["KERNELYRA_API_TOKEN"] = api_token
    else:
        environment.pop("KERNELYRA_API_TOKEN", None)
    return command, environment


class DaemonLock:
    """Exclusive workspace lock held for the complete daemon lifetime."""

    def __init__(self, workspace: str | Path, host: str, port: int):
        self.workspace = Path(workspace).expanduser().resolve()
        self.state_dir = self.workspace / ".kernelyra"
        self.host = host
        self.port = port
        self.handle: Any | None = None

    def __enter__(self) -> DaemonLock:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        ensure_user_secret(self.state_dir)
        ensure_agent_secret(self.state_dir)
        lock_path = self.state_dir / "daemon.lock"
        self.handle = lock_path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                getattr(fcntl, "flock")(
                    self.handle.fileno(), getattr(fcntl, "LOCK_EX") | getattr(fcntl, "LOCK_NB")
                )
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise ConfigurationError(
                f"Kernelyra daemon уже управляет workspace {self.workspace}"
            ) from exc
        (self.state_dir / "daemon.pid").write_text(str(os.getpid()), encoding="ascii")
        (self.state_dir / "daemon.json").write_text(
            json.dumps(
                {"pid": os.getpid(), "host": self.host, "port": self.port, "workspace": str(self.workspace)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return self

    def __exit__(self, *_: object) -> None:
        pid_path = self.state_dir / "daemon.pid"
        try:
            if pid_path.read_text(encoding="ascii").strip() == str(os.getpid()):
                pid_path.unlink(missing_ok=True)
        except OSError:
            pass
        if self.handle is not None:
            try:
                self.handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    getattr(fcntl, "flock")(self.handle.fileno(), getattr(fcntl, "LOCK_UN"))
            finally:
                self.handle.close()
                self.handle = None


def start_daemon(
    workspace: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    timeout: float = 20.0,
    api_token: str | None = None,
) -> dict[str, Any]:
    validate_daemon_bind(host, api_token)
    root = Path(workspace).expanduser().resolve()
    url = daemon_url(host, port)
    state_dir = root / ".kernelyra"
    user_secret = ensure_user_secret(state_dir)
    ensure_agent_secret(state_dir)
    client = DaemonClient(url, timeout=1.0, api_token=api_token, user_secret=user_secret)
    try:
        health = client.health()
    except DaemonUnavailableError:
        health = None
    if health:
        if Path(str(health.get("workspace", ""))).resolve() != root:
            raise ConfigurationError(
                f"Порт {host}:{port} уже занят Kernelyra для другого workspace: {health.get('workspace')}"
            )
        return {**health, "already_running": True, "url": url}

    log_path = state_dir / "daemon.log"
    log_handle = log_path.open("ab", buffering=0)
    command, child_environment = daemon_process_spec(root, host, port, url, api_token)
    # Fixed interpreter/module argv; shell execution is never enabled.
    process = subprocess.Popen(  # nosec B603
        command,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=root,
        env=child_environment,
        close_fds=os.name != "nt",
        **background_process_options(),
    )
    log_handle.close()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = ""
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            except OSError:
                pass
            raise ConfigurationError(f"Daemon завершился при запуске (код {process.returncode}).\n{tail}")
        try:
            health = client.health()
            return {**health, "already_running": False, "url": url}
        except DaemonUnavailableError:
            time.sleep(.15)
    process.terminate()
    raise DaemonUnavailableError(f"Daemon не ответил за {timeout:.0f} секунд. Лог: {log_path}")


def stop_daemon(workspace: str | Path, base_url: str, api_token: str | None = None) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    client = DaemonClient(
        base_url,
        timeout=2.0,
        api_token=api_token,
        user_secret=None if api_token else load_user_secret(root),
    )
    health = client.health()
    if Path(str(health.get("workspace", ""))).resolve() != root:
        raise ConfigurationError("Отказ остановки: daemon использует другой workspace")
    pid = int(health["pid"])
    if pid == os.getpid():
        raise ConfigurationError("Daemon нельзя остановить из его собственного процесса")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            client.health()
        except DaemonUnavailableError:
            return {"ok": True, "stopped_pid": pid, "workspace": str(root)}
        time.sleep(.15)
    raise ConfigurationError(f"Daemon PID {pid} не остановился за 10 секунд")
