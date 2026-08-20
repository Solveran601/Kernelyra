from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from kernelyra.errors import ConfigurationError
from kernelyra.security import (
    _ensure_secret,
    ensure_agent_secret,
    ensure_user_secret,
    load_agent_secret,
    load_user_secret,
    validate_daemon_bind,
)
from kernelyra.server import create_app
from tests.helpers import isolated_workspace


class SecurityTests(unittest.TestCase):
    def test_secret_creation_loading_permissions_and_race_are_safe(self) -> None:
        with isolated_workspace() as temporary:
            state = temporary / "project" / ".kernelyra"
            with patch("kernelyra.security.os.name", "posix"):
                user = ensure_user_secret(state)
            agent = ensure_agent_secret(state)
            self.assertGreaterEqual(len(user), 48)
            self.assertGreaterEqual(len(agent), 48)
            self.assertNotEqual(user, agent)
            self.assertEqual(ensure_user_secret(state), user)
            self.assertEqual(load_user_secret(temporary / "project"), user)
            self.assertEqual(load_agent_secret(temporary / "project"), agent)
            if os.name != "nt":
                self.assertEqual((state / "daemon.secret").stat().st_mode & 0o777, 0o600)

            with (
                patch.object(Path, "read_text", side_effect=[FileNotFoundError, "winner\n"]),
                patch("kernelyra.security.os.open", side_effect=FileExistsError),
            ):
                self.assertEqual(_ensure_secret(state, "raced.secret"), "winner")

    def test_missing_and_empty_local_secrets_are_rejected(self) -> None:
        with isolated_workspace() as temporary:
            project = temporary / "project"
            for loader in (load_user_secret, load_agent_secret):
                with self.subTest(loader=loader.__name__, state="missing"):
                    with self.assertRaises(ConfigurationError):
                        loader(project)
            state = project / ".kernelyra"
            state.mkdir(parents=True)
            (state / "daemon.secret").write_text("", encoding="ascii")
            (state / "agent.secret").write_text("\n", encoding="ascii")
            for loader in (load_user_secret, load_agent_secret):
                with self.subTest(loader=loader.__name__, state="empty"):
                    with self.assertRaises(ConfigurationError):
                        loader(project)

    def test_host_expiration_revocation_and_export_replay(self) -> None:
        with isolated_workspace() as temporary:
            project = temporary / "project"
            app = create_app(project)
            user_secret = (project / ".kernelyra" / "daemon.secret").read_text(encoding="ascii")
            agent_secret = (project / ".kernelyra" / "agent.secret").read_text(encoding="ascii")
            user = {"X-Kernelyra-User-Secret": user_secret}
            bootstrap = {"X-Kernelyra-Agent-Secret": agent_secret}
            with TestClient(app) as client:
                self.assertEqual(client.get("/api/v1/health", headers={"host": "attacker.invalid"}).status_code, 400)
                issued = client.post(
                    "/api/v1/mcp/sessions",
                    headers=bootstrap,
                    json={"client_id": "security-regression-agent", "ttl_seconds": 60},
                ).json()
                agent = {
                    "X-Kernelyra-Agent-Session": issued["session_token"],
                    "X-Kernelyra-Agent-Client": "security-regression-agent",
                }
                self.assertEqual(client.get("/api/v1/mcp/runs", headers=agent).status_code, 200)
                with patch("kernelyra.server.time.time", return_value=time.time() + 120):
                    self.assertEqual(client.get("/api/v1/mcp/runs", headers=agent).status_code, 403)

                issued = client.post(
                    "/api/v1/mcp/sessions",
                    headers=bootstrap,
                    json={"client_id": "security-regression-agent", "ttl_seconds": 60},
                ).json()
                agent["X-Kernelyra-Agent-Session"] = issued["session_token"]
                revoked = client.delete(f"/api/v1/agent-sessions/{issued['session_id']}", headers=user)
                self.assertTrue(revoked.json()["revoked"])
                self.assertEqual(client.get("/api/v1/mcp/runs", headers=agent).status_code, 403)

                run = client.post(
                    "/api/v1/runs",
                    headers=user,
                    json={"dataset": "demo", "backend": "numpy", "start": False},
                ).json()
                approval = client.post(
                    "/api/v1/approvals",
                    headers=user,
                    json={"action": "run.export", "resource_id": run["id"]},
                ).json()
                issued = client.post(
                    "/api/v1/mcp/sessions",
                    headers=bootstrap,
                    json={"client_id": "security-export-agent", "ttl_seconds": 60},
                ).json()
                export_agent = {
                    "X-Kernelyra-Agent-Session": issued["session_token"],
                    "X-Kernelyra-Agent-Client": "security-export-agent",
                }
                exported = client.post(
                    f"/api/v1/mcp/runs/{run['id']}/export",
                    headers=export_agent,
                    json={"approval_token": approval["approval_token"]},
                )
                self.assertEqual(exported.status_code, 200)
                self.assertNotIn(str(project), json.dumps(exported.json()))
                replay = client.post(
                    f"/api/v1/mcp/runs/{run['id']}/export",
                    headers=export_agent,
                    json={"approval_token": approval["approval_token"]},
                )
                self.assertEqual(replay.status_code, 403)
    def test_non_loopback_requires_strong_api_token(self) -> None:
        for host in ("0.0.0.0", "192.168.1.50", "::"):  # noqa: S104
            with self.assertRaises(ConfigurationError):
                validate_daemon_bind(host, None)
            with self.assertRaises(ConfigurationError):
                validate_daemon_bind(host, "short")
            validate_daemon_bind(host, "a" * 32)

    def test_loopback_does_not_require_network_token(self) -> None:
        for host in ("127.0.0.1", "localhost", "::1"):
            validate_daemon_bind(host, None)

    def test_only_minimal_health_is_public(self) -> None:
        with isolated_workspace() as temporary:
            project = temporary / "project"
            private_file = temporary / "private.csv"
            private_file.write_text("x,target\n1,0\n2,1\n", encoding="utf-8")
            app = create_app(project)
            secret = (project / ".kernelyra" / "daemon.secret").read_text(encoding="ascii")
            headers = {"X-Kernelyra-User-Secret": secret}
            with TestClient(app) as client:
                public_health = client.get("/api/v1/health")
                self.assertEqual(public_health.status_code, 200)
                self.assertEqual(set(public_health.json()), {"ok", "version", "protocol"})

                private_health = client.get("/api/v1/health", headers=headers).json()
                self.assertIn("pid", private_health)
                self.assertEqual(private_health["workspace"], str(project.resolve()))

                for path in (
                    "/api/v1/state",
                    "/api/v1/runs",
                    "/api/v1/datasets",
                    "/api/v1/logs",
                    "/api/v1/hardware",
                ):
                    denied = client.get(path)
                    self.assertEqual(denied.status_code, 403, path)
                    self.assertEqual(denied.json()["type"], "UserSessionRequired")

                inspect = client.post("/api/v1/paths/inspect", json={"path": str(private_file)})
                self.assertEqual(inspect.status_code, 403)
                root = client.get("/")
                self.assertEqual(root.status_code, 404)
                self.assertIn("terminal-first", root.json()["error"])

    def test_agent_credential_is_separate_and_least_privileged(self) -> None:
        with isolated_workspace() as temporary:
            project = temporary / "project"
            allowed = project / "allowed"
            allowed.mkdir(parents=True)
            allowed_file = allowed / "data.csv"
            allowed_file.write_text("x,target\n1,0\n", encoding="utf-8")
            outside_file = temporary / "outside.csv"
            outside_file.write_text("x,target\n1,0\n", encoding="utf-8")
            (project / "kernelyra.toml").write_text(
                '[mcp.permissions]\nallowed_roots = ["allowed"]\n'
                'allowed_actions = ["path.inspect", "run.create", "run.list", "run.pause", '
                '"run.stop", "dataset.list", "hardware.read"]\n',
                encoding="utf-8",
            )
            app = create_app(project)
            user_secret = (project / ".kernelyra" / "daemon.secret").read_text(encoding="ascii")
            agent_secret = (project / ".kernelyra" / "agent.secret").read_text(encoding="ascii")
            self.assertNotEqual(user_secret, agent_secret)
            user = {"X-Kernelyra-User-Secret": user_secret}
            bootstrap = {"X-Kernelyra-Agent-Secret": agent_secret}
            run_body = {"dataset": "demo", "backend": "numpy", "start": False, "max_steps": 10000}
            with TestClient(app) as client:
                denied_draft = client.post("/api/v1/mcp/runs", json=run_body)
                self.assertEqual(denied_draft.status_code, 403)
                self.assertEqual(denied_draft.json()["type"], "AgentSessionRequired")

                direct_secret = client.post("/api/v1/mcp/runs", headers=bootstrap, json=run_body)
                self.assertEqual(direct_secret.status_code, 403)
                issued = client.post(
                    "/api/v1/mcp/sessions",
                    headers=bootstrap,
                    json={"client_id": "security-test-agent", "ttl_seconds": 120},
                )
                self.assertEqual(issued.status_code, 201)
                self.assertEqual(issued.json()["allowed_roots"], [str(allowed.resolve())])
                self.assertNotIn("logs.read", issued.json()["allowed_actions"])
                agent = {
                    "X-Kernelyra-Agent-Session": issued.json()["session_token"],
                    "X-Kernelyra-Agent-Client": "security-test-agent",
                }

                draft = client.post("/api/v1/mcp/runs", headers=agent, json=run_body)
                self.assertEqual(draft.status_code, 201)
                self.assertEqual(draft.json()["status"], "draft")

                for path in (
                    "/api/v1/mcp/runs",
                    "/api/v1/mcp/datasets",
                    "/api/v1/mcp/hardware",
                ):
                    self.assertEqual(client.get(path).status_code, 403, path)
                    self.assertEqual(client.get(path, headers=agent).status_code, 200, path)
                self.assertEqual(client.get("/api/v1/mcp/logs", headers=agent).status_code, 403)
                wrong_client = {
                    "X-Kernelyra-Agent-Session": issued.json()["session_token"],
                    "X-Kernelyra-Agent-Client": "different-agent-client",
                }
                self.assertEqual(client.get("/api/v1/mcp/runs", headers=wrong_client).status_code, 403)

                denied_agent_inspect = client.post(
                    "/api/v1/mcp/paths/inspect", json={"path": str(allowed_file)}
                )
                self.assertEqual(denied_agent_inspect.status_code, 403)
                self.assertEqual(
                    client.post(
                        "/api/v1/mcp/paths/inspect",
                        headers=agent,
                        json={"path": str(allowed_file)},
                    ).status_code,
                    200,
                )
                outside = client.post(
                    "/api/v1/mcp/paths/inspect",
                    headers=agent,
                    json={"path": str(outside_file)},
                )
                self.assertEqual(outside.status_code, 403)

                self.assertEqual(client.get("/api/v1/state", headers=agent).status_code, 403)
                self.assertEqual(client.get("/api/v1/state", headers=bootstrap).status_code, 403)
                self.assertEqual(client.post("/api/v1/runs", headers=agent, json=run_body).status_code, 403)
                self.assertEqual(
                    client.post(
                        "/api/v1/approvals",
                        headers=agent,
                        json={"action": "run.start", "resource_id": draft.json()["id"]},
                    ).status_code,
                    403,
                )

                pausable = client.post("/api/v1/runs", headers=user, json=run_body).json()
                client.post(
                    f"/api/v1/runs/{pausable['id']}/command",
                    headers=user,
                    json={"command": "start"},
                )
                for command in ("pause", "stop"):
                    denied = client.post(
                        f"/api/v1/mcp/runs/{pausable['id']}/command",
                        json={"command": command, "approval_token": ""},
                    )
                    self.assertEqual(denied.status_code, 403, command)
                allowed_pause = client.post(
                    f"/api/v1/mcp/runs/{pausable['id']}/command",
                    headers=agent,
                    json={"command": "pause", "approval_token": ""},
                )
                self.assertEqual(allowed_pause.status_code, 200)

                stoppable = client.post("/api/v1/runs", headers=user, json=run_body).json()
                client.post(
                    f"/api/v1/runs/{stoppable['id']}/command",
                    headers=user,
                    json={"command": "start"},
                )
                allowed_stop = client.post(
                    f"/api/v1/mcp/runs/{stoppable['id']}/command",
                    headers=agent,
                    json={"command": "stop", "approval_token": ""},
                )
                self.assertEqual(allowed_stop.status_code, 200)

                with patch(
                    "kernelyra.server.time.time",
                    return_value=float(issued.json()["expires_at"]) + 1,
                ):
                    expired = client.get("/api/v1/mcp/runs", headers=agent)
                self.assertEqual(expired.status_code, 403)
                self.assertEqual(expired.json()["type"], "AgentSessionRequired")

    def test_configured_api_token_is_user_access_but_not_agent_access(self) -> None:
        with isolated_workspace() as temporary:
            project = temporary / "project"
            token = "network-token-" + "x" * 32
            app = create_app(project, api_token=token)
            agent_secret = (project / ".kernelyra" / "agent.secret").read_text(encoding="ascii")
            bearer = {"Authorization": f"Bearer {token}"}
            with TestClient(app) as client:
                self.assertEqual(client.get("/api/v1/health").status_code, 200)
                self.assertEqual(set(client.get("/api/v1/health").json()), {"ok", "version", "protocol"})
                self.assertEqual(client.get("/api/v1/state").status_code, 401)
                self.assertEqual(client.get("/api/v1/state", headers=bearer).status_code, 200)
                self.assertEqual(client.get("/api/v1/mcp/runs", headers=bearer).status_code, 403)
                bootstrap = {**bearer, "X-Kernelyra-Agent-Secret": agent_secret}
                session = client.post(
                    "/api/v1/mcp/sessions",
                    headers=bootstrap,
                    json={"client_id": "network-test-agent"},
                )
                self.assertEqual(session.status_code, 201)
                agent_headers = {
                    **bearer,
                    "X-Kernelyra-Agent-Session": session.json()["session_token"],
                    "X-Kernelyra-Agent-Client": "network-test-agent",
                }
                self.assertEqual(client.get("/api/v1/mcp/runs", headers=agent_headers).status_code, 200)

    def test_root_is_headless_in_network_mode(self) -> None:
        with isolated_workspace() as temporary:
            token = "network-token-" + "x" * 32
            app = create_app(temporary / "project", api_token=token, enable_ui=False)
            with TestClient(app) as client:
                response = client.get("/")
                self.assertEqual(response.status_code, 404)
                self.assertIn("terminal-first", response.json()["error"])

    def test_unauthorized_http_cannot_import_path_or_start_training(self) -> None:
        with isolated_workspace() as temporary:
            project = temporary / "project"
            source = temporary / "private.csv"
            source.write_text("x,target\n" + "\n".join(f"{i},{i % 2}" for i in range(50)), encoding="utf-8")
            app = create_app(project)
            secret = (project / ".kernelyra" / "daemon.secret").read_text(encoding="ascii")
            headers = {"X-Kernelyra-User-Secret": secret}
            with TestClient(app) as client:
                denied_import = client.post(
                    "/api/v1/datasets/from-path", json={"path": str(source), "target": "target"}
                )
                self.assertEqual(denied_import.status_code, 403)
                self.assertEqual(denied_import.json()["type"], "UserSessionRequired")

                created = client.post(
                    "/api/v1/runs",
                    headers=headers,
                    json={"dataset": "demo", "backend": "numpy", "start": False},
                ).json()
                denied_start = client.post(
                    f"/api/v1/runs/{created['id']}/command", json={"command": "start"}
                )
                self.assertEqual(denied_start.status_code, 403)
                self.assertEqual(app.state.workspace.storage.get_run(created["id"]).status, "draft")

                allowed_start = client.post(
                    f"/api/v1/runs/{created['id']}/command",
                    headers=headers,
                    json={"command": "start"},
                )
                self.assertEqual(allowed_start.status_code, 200)

if __name__ == "__main__":
    unittest.main()
