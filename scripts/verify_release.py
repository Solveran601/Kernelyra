from __future__ import annotations

import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))

from check_clean_source import forbidden_name  # noqa: E402


def _python_in(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _script_in(venv: Path, name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return venv / ("Scripts" if os.name == "nt" else "bin") / f"{name}{suffix}"


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _checked(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, encoding="utf-8", capture_output=True)
    if result.returncode:
        raise SystemExit(
            f"Installed-wheel command failed ({result.returncode}): {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def _expect_http_denied(
    url: str,
    path: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> None:
    call = request.Request(  # noqa: S310 - verifier uses a generated loopback HTTP origin
        url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        request.urlopen(call, timeout=5)  # noqa: S310 - generated loopback HTTP origin
    except urlerror.HTTPError as error:
        if error.code == 403:
            return
        raise SystemExit(f"Expected HTTP 403 for {path}, received {error.code}") from None
    raise SystemExit(f"Unauthenticated installed-wheel request unexpectedly succeeded: {path}")


def _expect_get_denied(url: str, path: str, headers: dict[str, str] | None = None) -> None:
    call = request.Request(url + path, headers=headers or {}, method="GET")  # noqa: S310
    try:
        request.urlopen(call, timeout=5)  # noqa: S310 - generated loopback HTTP origin
    except urlerror.HTTPError as error:
        if error.code == 403:
            return
        raise SystemExit(f"Expected HTTP 403 for {path}, received {error.code}") from None
    raise SystemExit(f"Unauthenticated installed-wheel GET unexpectedly succeeded: {path}")


def _expect_get_status(url: str, path: str, status: int) -> None:
    call = request.Request(url + path, method="GET")  # noqa: S310
    try:
        with request.urlopen(call, timeout=5) as response:  # noqa: S310
            received = response.status
    except urlerror.HTTPError as error:
        received = error.code
    if received != status:
        raise SystemExit(f"Expected HTTP {status} for {path}, received {received}")


def _json_request(
    url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    opener: Any | None = None,
) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    call_headers = {"Accept": "application/json", **(headers or {})}
    if body is not None:
        call_headers["Content-Type"] = "application/json"
    call = request.Request(url + path, data=body, headers=call_headers, method=method)  # noqa: S310
    open_call = request.urlopen if opener is None else opener.open
    with open_call(call, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


class RawMCPClient:
    """Small JSON-RPC client proving the installed console script's stdio transport."""

    def __init__(self, command: list[str], cwd: Path, env: dict[str, str]):
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self.reader = threading.Thread(target=self._read, name="wheel-mcp-reader", daemon=True)
        self.reader.start()
        self.next_id = 1

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            try:
                self.messages.put(json.loads(line))
            except json.JSONDecodeError:
                continue

    def send(self, message: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_id = self.next_id
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                message = self.messages.get(timeout=max(.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            if message.get("id") == request_id:
                if "error" in message:
                    raise SystemExit(f"Installed-wheel MCP error for {method}: {message['error']}")
                return message["result"]
        raise SystemExit(f"Installed-wheel MCP timeout for {method}; process={self.process.poll()}")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self.send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)
        if self.process.stdout:
            self.process.stdout.close()


def archive_names(path: Path) -> list[str]:
    if path.suffix in {".whl", ".zip"}:
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    with tarfile.open(path, "r:gz") as archive:
        return archive.getnames()


def forbidden_archive_entry(artifact: Path, name: str) -> bool:
    if artifact.name.endswith(".tar.gz"):
        parts = [part for part in name.replace("\\", "/").split("/") if part]
        without_standard_metadata = [
            part for part in parts if not part.lower().endswith(".egg-info")
        ]
        return forbidden_name("/".join(without_standard_metadata))
    return forbidden_name(name)


def main() -> int:
    dist = ROOT / "dist"
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    source_zips = sorted(dist.glob("*-source.zip"))
    if len(wheels) != 1 or len(sdists) != 1 or len(source_zips) != 1:
        raise SystemExit("Expected exactly one wheel, one sdist and one source ZIP in dist/")
    for artifact in [*wheels, *sdists, *source_zips]:
        forbidden = [
            name for name in archive_names(artifact) if forbidden_archive_entry(artifact, name)
        ]
        if forbidden:
            raise SystemExit(f"Forbidden files in {artifact.name}:\n" + "\n".join(forbidden))
    sdist_names = archive_names(sdists[0])
    distributable_sources = (
        "native/core/zig/memory_kernels.zig",
        "native/core/fortran/training_kernels.f90",
        "native/include/kernelyra_core.h",
        "src/kernelyra/formats.py",
        "src/kernelyra/architectures.py",
        "sdks/go/go.mod",
        "sdks/cpp/CMakeLists.txt",
        "sdks/csharp/Kernelyra.Client.csproj",
        "sdks/csharp/Directory.Build.props",
        "sdks/rust/Cargo.toml",
    )
    for required in (
        "worker.py",
        "start_worker.bat",
        "start_worker.ps1",
        "requirements.txt",
        "LICENSE",
        *distributable_sources,
    ):
        if not any(name.endswith("/" + required) for name in sdist_names):
            raise SystemExit(f"Required launcher/source file is missing from sdist: {required}")
    source_names = archive_names(source_zips[0])
    for required in (
        "scripts/check_clean_source.py",
        "scripts/build_source_bundle.py",
        "LICENSE",
        *distributable_sources,
    ):
        if not any(name.endswith("/" + required) for name in source_names):
            raise SystemExit(f"Required file is missing from source ZIP: {required}")

    verify_root = ROOT / "build" / f"release-verify-{uuid.uuid4().hex}"
    venv = verify_root / "venv"
    work = verify_root / "empty-workspace"
    daemon_workspace = verify_root / "daemon-workspace"
    daemon_started = False
    daemon_process: subprocess.Popen[bytes] | None = None
    kernelyra: Path | None = None
    smoke_env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    smoke_env.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, cwd=ROOT)
        python = _python_in(venv)
        requirement = f"kernelyra-ai[gateway,mcp,parquet] @ {wheels[0].resolve().as_uri()}"
        subprocess.run(
            [
                str(python), "-m", "pip", "install",
                "-c", str(ROOT / "constraints" / "release.txt"), requirement,
            ],
            check=True,
            cwd=verify_root,
            env=smoke_env,
        )
        work.mkdir(parents=True)
        code = r'''
import json, os, sys, threading
before_threads = {thread.name for thread in threading.enumerate()}
before_files = set(os.listdir('.'))
import kernelyra
after_threads = {thread.name for thread in threading.enumerate()}
payload = {
    'version': kernelyra.__version__,
    'tensorflow_loaded': 'tensorflow' in sys.modules,
    'new_threads': sorted(after_threads - before_threads),
    'new_files': sorted(set(os.listdir('.')) - before_files),
}
print(json.dumps(payload, sort_keys=True))
assert payload['version'] == '0.4.0a1'
assert payload['tensorflow_loaded'] is False
assert payload['new_threads'] == []
assert payload['new_files'] == []
'''
        result = subprocess.run(
            [str(python), "-c", code],
            check=True,
            text=True,
            capture_output=True,
            cwd=work,
            env=smoke_env,
        )
        payload = json.loads(result.stdout)

        kernelyra = _script_in(venv, "kernelyra")
        daemon_workspace.mkdir()

        def cli(*arguments: str) -> subprocess.CompletedProcess[str]:
            return _checked(
                [str(kernelyra), "--workspace", str(daemon_workspace), "--daemon-url", url, *arguments],
                cwd=work,
                env=smoke_env,
            )

        if cli("version").stdout.strip() != "0.4.0a1":
            raise SystemExit("Installed-wheel CLI reported an unexpected version")
        doctor = json.loads(cli("--json", "doctor").stdout)
        if not doctor.get("ok") or doctor.get("version") != "0.4.0a1":
            raise SystemExit(f"Installed-wheel doctor failed: {doctor}")
        capabilities = json.loads(cli("--json", "capabilities").stdout)
        if not any(item.get("name") == "numpy" and item.get("available") for item in capabilities["backends"]):
            raise SystemExit(f"Installed-wheel NumPy capability is unavailable: {capabilities}")
        if capabilities.get("extensions_enabled") is not False:
            raise SystemExit(f"Installed wheel unexpectedly enables dynamic extensions: {capabilities}")

        daemon_process = subprocess.Popen(
            [
                str(kernelyra), "--workspace", str(daemon_workspace), "--daemon-url", url,
                "daemon", "foreground", "--port", str(port),
            ],
            cwd=work,
            env=smoke_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if daemon_process.poll() is not None:
                raise SystemExit(f"Installed-wheel daemon exited with code {daemon_process.returncode}")
            try:
                with request.urlopen(url + "/api/v1/health", timeout=.5) as response:  # noqa: S310
                    if response.status == 200:
                        daemon_started = True
                        break
            except OSError:
                time.sleep(.1)
        if not daemon_started:
            raise SystemExit("Installed-wheel daemon did not become healthy")
        formats = json.loads(cli("--json", "formats").stdout)
        if formats.get("recognized_routes") != 535 or not {".csv", ".parquet"}.issubset(
            formats.get("trainable_extensions", [])
        ):
            raise SystemExit(f"Installed-wheel terminal format registry failed: {formats}")
        public_health = _json_request(url, "/api/v1/health")
        if set(public_health) != {"ok", "version", "protocol"}:
            raise SystemExit(f"Public health leaked private fields: {public_health}")
        health = json.loads(cli("daemon", "status").stdout)
        if health.get("version") != "0.4.0a1" or Path(health["workspace"]).resolve() != daemon_workspace.resolve():
            raise SystemExit(f"Unexpected installed-wheel daemon health: {health}")

        _expect_get_status(url, "/", 404)
        for private_path in (
            "/api/v1/state",
            "/api/v1/runs",
            "/api/v1/datasets",
            "/api/v1/logs",
            "/api/v1/hardware",
            "/api/v1/mcp/runs",
        ):
            _expect_get_denied(url, private_path)

        unapproved = work / "unapproved.csv"
        unapproved.write_text("x,target\n1,0\n2,1\n", encoding="utf-8")
        _expect_http_denied(url, "/api/v1/paths/inspect", {"path": str(unapproved)})
        _expect_http_denied(url, "/api/v1/mcp/paths/inspect", {"path": str(unapproved)})
        _expect_http_denied(
            url,
            "/api/v1/mcp/runs",
            {"dataset": "demo", "backend": "numpy", "start": False},
        )

        agent_secret = (daemon_workspace / ".kernelyra" / "agent.secret").read_text(encoding="ascii").strip()
        bootstrap_headers = {"X-Kernelyra-Agent-Secret": agent_secret}
        _expect_http_denied(
            url,
            "/api/v1/datasets/from-path",
            {"path": str(unapproved), "target": "target"},
        )
        _expect_get_denied(url, "/api/v1/mcp/runs", headers=bootstrap_headers)
        scoped = _json_request(
            url,
            "/api/v1/mcp/sessions",
            method="POST",
            payload={"client_id": "release-verifier-agent", "ttl_seconds": 300},
            headers=bootstrap_headers,
        )
        agent_headers = {
            "X-Kernelyra-Agent-Session": scoped["session_token"],
            "X-Kernelyra-Agent-Client": "release-verifier-agent",
        }
        if not isinstance(_json_request(url, "/api/v1/mcp/runs", headers=agent_headers), list):
            raise SystemExit("Scoped agent session could not read the permitted agent route")
        _expect_http_denied(
            url,
            "/api/v1/mcp/paths/inspect",
            {"path": str(unapproved)},
            headers=agent_headers,
        )
        _expect_get_denied(url, "/api/v1/state", headers=agent_headers)
        _expect_http_denied(
            url,
            "/api/v1/approvals",
            {"action": "run.start", "resource_id": "forbidden"},
            headers=agent_headers,
        )

        created = json.loads(
            cli(
                "run", "create", "--dataset", "demo", "--backend", "numpy",
                "--target-metric", "1.0", "--max-steps", "1000000",
            ).stdout
        )
        run_id = created["id"]
        _expect_http_denied(
            url,
            f"/api/v1/runs/{run_id}/command",
            {"command": "start"},
        )
        for command in ("pause", "stop"):
            _expect_http_denied(
                url,
                f"/api/v1/mcp/runs/{run_id}/command",
                {"command": command, "approval_token": ""},
            )

        def wait_run(expected: str, *, minimum_step: int = 0) -> dict[str, Any]:
            deadline = time.monotonic() + 20
            state: dict[str, Any] = {}
            while time.monotonic() < deadline:
                state = json.loads(cli("run", "get", run_id).stdout)
                if state["status"] == expected and int(state["step"]) >= minimum_step:
                    return state
                if state["status"] in {"completed", "error_recoverable"}:
                    break
                time.sleep(.1)
            raise SystemExit(
                f"Installed-wheel run did not reach {expected} at step {minimum_step}: {state}"
            )

        cli("run", "start", run_id)
        first_training = wait_run("training", minimum_step=20)
        paused = json.loads(cli("run", "pause", run_id).stdout)
        if paused["status"] != "paused":
            raise SystemExit(f"Pause was not acknowledged by the installed-wheel worker: {paused}")

        cli("run", "resume", run_id)
        wait_run("training", minimum_step=int(first_training["step"]) + 20)
        stopped = json.loads(cli("run", "stop", run_id).stdout)
        if stopped["status"] != "stopped":
            raise SystemExit(f"Unexpected installed-wheel stop state: {stopped['status']}")

        allowed_dataset = daemon_workspace / "allowed.csv"
        allowed_dataset.write_text("x,target\n1,0\n2,1\n", encoding="utf-8")
        mcp = RawMCPClient(
            [str(kernelyra), "--workspace", str(daemon_workspace), "--daemon-url", url, "mcp"],
            work,
            smoke_env,
        )
        try:
            initialized = mcp.request(
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "kernelyra-wheel-verifier", "version": "1"},
                },
            )
            if initialized["serverInfo"]["name"] != "Kernelyra":
                raise SystemExit(f"Unexpected installed-wheel MCP server: {initialized}")
            mcp.notify("notifications/initialized")
            tools = mcp.request("tools/list")
            tool_names = {item["name"] for item in tools["tools"]}
            if "kernelyra.list_runs" not in tool_names:
                raise SystemExit("Installed-wheel MCP tools/list is incomplete")
            listed = mcp.request("tools/call", {"name": "kernelyra.list_runs", "arguments": {}})
            if listed.get("isError"):
                raise SystemExit(f"Installed-wheel MCP tools/call failed: {listed}")
            inspected = mcp.request(
                "tools/call",
                {"name": "kernelyra.inspect_path", "arguments": {"path": "allowed.csv"}},
            )
            if inspected.get("isError"):
                raise SystemExit(f"Installed-wheel scoped MCP path inspection failed: {inspected}")
            resources = mcp.request("resources/list")
            resource_uris = {item["uri"] for item in resources["resources"]}
            if "kernelyra://runs" not in resource_uris:
                raise SystemExit("Installed-wheel MCP resources/list is incomplete")
            read = mcp.request("resources/read", {"uri": "kernelyra://runs"})
            if not read.get("contents"):
                raise SystemExit("Installed-wheel MCP resources/read returned no content")
        finally:
            mcp.close()

        print("Release archive check: OK")
        print("Clean wheel install: OK")
        print("Installed wheel doctor/capabilities/closed registries: OK")
        print("Installed wheel terminal format registry: OK")
        print("Installed wheel headless daemon/CLI lifecycle: OK")
        print("Installed wheel unauthenticated mutation guard: OK")
        print("Installed wheel server-side scoped path guard: OK")
        print("Installed wheel MCP stdio E2E: OK")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        if daemon_started and kernelyra is not None:
            subprocess.run(
                [str(kernelyra), "--workspace", str(daemon_workspace), "--daemon-url", url, "daemon", "stop"],
                cwd=work if work.exists() else verify_root,
                env=smoke_env,
                text=True,
                encoding="utf-8",
                capture_output=True,
            )
        if daemon_process is not None and daemon_process.poll() is None:
            daemon_process.terminate()
            try:
                daemon_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                daemon_process.kill()
                daemon_process.wait(timeout=5)
        shutil.rmtree(verify_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
