from __future__ import annotations

import contextlib
import io
import os
import socket
import subprocess
import sys
import time
import unittest

from kernelyra.cli import main as cli_main
from kernelyra.client import DaemonClient, RemoteError
from kernelyra.models import RunConfig
from tests.helpers import ROOT, isolated_workspace


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class DaemonIntegrationTests(unittest.TestCase):
    def test_cli_mcp_approval_lock_and_multiple_clients_share_one_runtime(self) -> None:
        with isolated_workspace() as temporary:
            workspace = temporary / "project"
            port = free_port()
            other_port = free_port()
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
            try:
                first = DaemonClient(url, timeout=.5)
                deadline = time.monotonic() + 20
                while True:
                    try:
                        health = first.health()
                        break
                    except Exception:
                        if daemon.poll() is not None or time.monotonic() >= deadline:
                            self.fail("daemon did not start")
                        time.sleep(.1)
                self.assertEqual(health["protocol"], "kernelyra-api/v1")
                user_secret = (workspace / ".kernelyra" / "daemon.secret").read_text(encoding="ascii")
                agent_secret = (workspace / ".kernelyra" / "agent.secret").read_text(encoding="ascii")
                # A cold Windows worker may spend more than 500 ms loading the
                # local runtime on low-power hardware. The production default
                # is already longer; keep the integration gate realistic too.
                first = DaemonClient(url, timeout=2.0, user_secret=user_secret)
                second = DaemonClient(url, user_secret=user_secret)
                agent = DaemonClient(url, agent_secret=agent_secret)
                self.assertEqual(first.health()["engine"], "single-daemon")
                run = first.create_run(
                    RunConfig(dataset="demo", backend="numpy", max_steps=10000, target_metric=.999)
                )
                common = ["--workspace", str(workspace), "--daemon-url", url, "run"]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli_main([*common, "start", run["id"]]), 0)
                    duplicate = second.command(run["id"], "start")
                    self.assertIn(duplicate["status"], {"queued", "training"})
                    self.assertEqual(cli_main([*common, "pause", run["id"]]), 0)
                    self.assertEqual(cli_main([*common, "resume", run["id"]]), 0)
                    self.assertEqual(cli_main([*common, "stop", run["id"]]), 0)
                stop_deadline = time.monotonic() + 10
                while first.get_run(run["id"])["status"] != "stopped":
                    if time.monotonic() >= stop_deadline:
                        self.fail("run did not finish stopping")
                    time.sleep(.05)

                approved_run = second.create_run(RunConfig(dataset="demo", backend="numpy"))
                with self.assertRaises(RemoteError) as missing:
                    agent.mcp_command(approved_run["id"], "start", "not-a-token")
                self.assertEqual(missing.exception.status, 403)
                with self.assertRaises(RemoteError) as unauthenticated_approval:
                    DaemonClient(url).issue_approval("run.start", approved_run["id"])
                self.assertEqual(unauthenticated_approval.exception.status, 403)
                approval = first.issue_approval("run.start", approved_run["id"], user_secret=user_secret)
                started = agent.mcp_command(approved_run["id"], "start", approval["approval_token"])
                self.assertIn(started["status"], {"queued", "training"})
                with self.assertRaises(RemoteError) as reused:
                    agent.mcp_command(approved_run["id"], "start", approval["approval_token"])
                self.assertEqual(reused.exception.status, 403)
                second.command(approved_run["id"], "stop")
                actions = first.logs(100)
                self.assertTrue(any(item["action"] == "approval.consume" for item in actions))
                issued = next(item for item in actions if item["action"] == "approval.issue")
                self.assertEqual(issued["actor"], "user_cli")

                second_daemon = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "kernelyra.cli",
                        "--workspace",
                        str(workspace),
                        "daemon",
                        "foreground",
                        "--port",
                        str(other_port),
                    ],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=15,
                )
                self.assertEqual(second_daemon.returncode, 2)
                self.assertIn("уже управляет workspace", second_daemon.stderr)
            finally:
                if daemon.poll() is None:
                    daemon.terminate()
                    try:
                        daemon.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        daemon.kill()
                        daemon.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
