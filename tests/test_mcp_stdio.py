from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from kernelyra.client import DaemonClient
from kernelyra.models import RunConfig
from tests.helpers import ROOT, isolated_workspace


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class RawMCPClient:
    """Minimal JSON-RPC MCP client using the server's real stdio transport."""

    def __init__(self, command: list[str], env: dict[str, str], cwd: Path = ROOT):
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
        self.reader = threading.Thread(target=self._read, name="mcp-test-reader", daemon=True)
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
                    raise AssertionError(f"MCP error for {method}: {message['error']}")
                return message["result"]
        raise AssertionError(f"MCP timeout for {method}; process={self.process.poll()}")

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


class MCPStdioTests(unittest.TestCase):
    def test_real_stdio_handshake_tools_resources_allowlist_and_approval(self) -> None:
        with isolated_workspace() as temporary:
            workspace = temporary / "project"
            workspace.mkdir()
            allowed = workspace / "allowed.csv"
            allowed.write_text("x,target\n" + "\n".join(f"{i},{i % 2}" for i in range(50)), encoding="utf-8")
            outside = temporary / "outside.csv"
            outside.write_text(allowed.read_text(encoding="utf-8"), encoding="utf-8")
            config = workspace / "kernelyra.toml"
            config.write_text('[mcp.permissions]\nallowed_roots = ["."]\n', encoding="utf-8")
            foreign_cwd = temporary / "foreign-cwd"
            foreign_cwd.mkdir()
            port = free_port()
            url = f"http://127.0.0.1:{port}"
            env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
            daemon = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "kernelyra.cli",
                    "--workspace",
                    str(workspace),
                    "daemon",
                    "foreground",
                    "--port",
                    str(port),
                ],
                cwd=ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            mcp: RawMCPClient | None = None
            try:
                client = DaemonClient(url, timeout=.5)
                deadline = time.monotonic() + 20
                while True:
                    try:
                        client.health()
                        break
                    except Exception:
                        if daemon.poll() is not None or time.monotonic() >= deadline:
                            self.fail("daemon did not start")
                        time.sleep(.1)
                secret = (workspace / ".kernelyra" / "daemon.secret").read_text(encoding="ascii")
                client = DaemonClient(url, timeout=.5, user_secret=secret)
                run = client.create_run(RunConfig(dataset="demo", backend="numpy", max_steps=10000))
                approval = client.issue_approval("run.start", run["id"], user_secret=secret)

                mcp = RawMCPClient(
                    [
                        sys.executable,
                        "-m",
                        "kernelyra.cli",
                        "--workspace",
                        str(workspace),
                        "--daemon-url",
                        url,
                        "mcp",
                        "--config",
                        str(config),
                    ],
                    env,
                    cwd=foreign_cwd,
                )
                initialized = mcp.request(
                    "initialize",
                    {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "kernelyra-e2e-test", "version": "1"},
                    },
                )
                self.assertEqual(initialized["serverInfo"]["name"], "Kernelyra")
                mcp.notify("notifications/initialized")

                tools = mcp.request("tools/list")
                names = {tool["name"] for tool in tools["tools"]}
                self.assertIn("kernelyra.start_run", names)
                self.assertIn("kernelyra.inspect_path", names)

                allowed_result = mcp.request(
                    "tools/call", {"name": "kernelyra.inspect_path", "arguments": {"path": allowed.name}}
                )
                self.assertFalse(allowed_result.get("isError", False))
                denied_result = mcp.request(
                    "tools/call", {"name": "kernelyra.inspect_path", "arguments": {"path": str(outside)}}
                )
                self.assertTrue(denied_result.get("isError", False))

                started = mcp.request(
                    "tools/call",
                    {
                        "name": "kernelyra.start_run",
                        "arguments": {"run_id": run["id"], "approval_token": approval["approval_token"]},
                    },
                )
                self.assertFalse(started.get("isError", False))
                reused = mcp.request(
                    "tools/call",
                    {
                        "name": "kernelyra.start_run",
                        "arguments": {"run_id": run["id"], "approval_token": approval["approval_token"]},
                    },
                )
                self.assertTrue(reused.get("isError", False))
                stopped = mcp.request(
                    "tools/call", {"name": "kernelyra.stop_run", "arguments": {"run_id": run["id"]}}
                )
                self.assertFalse(stopped.get("isError", False))

                resources = mcp.request("resources/list")
                uris = {resource["uri"] for resource in resources["resources"]}
                self.assertIn("kernelyra://runs", uris)
                read_result = mcp.request("resources/read", {"uri": "kernelyra://runs"})
                self.assertTrue(read_result["contents"])
            finally:
                if mcp:
                    mcp.close()
                if daemon.poll() is None:
                    daemon.terminate()
                    try:
                        daemon.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        daemon.kill()
                        daemon.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
