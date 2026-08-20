from __future__ import annotations

import unittest
from pathlib import Path

from kernelyra.daemon import background_process_options, daemon_process_spec, daemon_url


class DaemonProcessOptionsTests(unittest.TestCase):
    def test_posix_background_daemon_starts_a_new_session(self) -> None:
        self.assertEqual(background_process_options("posix"), {"start_new_session": True})

    def test_windows_background_daemon_uses_detached_creation_flags(self) -> None:
        options = background_process_options("nt")
        self.assertNotIn("start_new_session", options)
        self.assertGreater(options["creationflags"], 0)

    def test_ipv6_daemon_url_uses_brackets(self) -> None:
        self.assertEqual(daemon_url("::1", 18766), "http://[::1]:18766")
        self.assertEqual(daemon_url("127.0.0.1", 8765), "http://127.0.0.1:8765")

    def test_network_token_is_in_child_environment_not_argv(self) -> None:
        token = "secret-network-token-" + "x" * 24
        command, environment = daemon_process_spec(
            Path("project"), "0.0.0.0", 8765, "http://0.0.0.0:8765", token  # noqa: S104
        )
        self.assertNotIn(token, command)
        self.assertNotIn("--api-token", command)
        self.assertEqual(environment["KERNELYRA_API_TOKEN"], token)


if __name__ == "__main__":
    unittest.main()
