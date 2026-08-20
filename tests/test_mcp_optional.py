from __future__ import annotations

import builtins
import unittest
from unittest.mock import patch

from kernelyra.errors import ConfigurationError
from kernelyra.mcp_server import build_mcp


class MCPOptionalDependencyTests(unittest.TestCase):
    def test_missing_mcp_reports_install_command_without_import_traceback(self) -> None:
        real_import = builtins.__import__

        def import_without_mcp(name, *args, **kwargs):
            if name == "mcp.server.fastmcp":
                error = ModuleNotFoundError("No module named 'mcp'")
                error.name = "mcp"
                raise error
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_mcp):
            with self.assertRaisesRegex(ConfigurationError, r'pip install "kernelyra-ai\[mcp\]"'):
                build_mcp(".")


if __name__ == "__main__":
    unittest.main()
