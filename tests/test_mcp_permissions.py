from __future__ import annotations

import os
import unittest

from kernelyra.mcp_server import MCPPermissions
from tests.helpers import isolated_workspace


class MCPPermissionPathTests(unittest.TestCase):
    def test_relative_tool_path_uses_config_directory_not_process_cwd(self) -> None:
        with isolated_workspace() as temporary:
            workspace = temporary / "project"
            workspace.mkdir()
            dataset = workspace / "data.csv"
            dataset.write_text("x,target\n1,0\n", encoding="utf-8")
            config = workspace / "kernelyra.toml"
            config.write_text('[mcp.permissions]\nallowed_roots = ["."]\n', encoding="utf-8")
            foreign = temporary / "foreign"
            foreign.mkdir()
            previous = os.getcwd()
            try:
                os.chdir(foreign)
                permissions = MCPPermissions.load(workspace, config)
                self.assertEqual(permissions.require_path("data.csv"), dataset.resolve())
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
