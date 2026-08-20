from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from unittest.mock import patch

from kernelyra import Workspace
from kernelyra.ingestion.router import FormatRouter
from tests.helpers import ROOT, isolated_workspace


class NativeProbeTests(unittest.TestCase):
    def test_cpp_json_contract_matches_python_router(self) -> None:
        compiler = shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            self.skipTest("C++17 compiler is not installed")
        with isolated_workspace() as temporary:
            executable = temporary / ("dataset_probe.exe" if __import__("os").name == "nt" else "dataset_probe")
            subprocess.run(
                [compiler, "-std=c++17", "-O2", str(ROOT / "native" / "tools" / "dataset_probe.cpp"), "-o", str(executable)],
                check=True,
                timeout=120,
            )
            sample = temporary / "sample.csv"
            sample.write_text("x,target\n1,0\n", encoding="utf-8")
            result = subprocess.run(
                [str(executable), "--json", str(sample)],
                text=True,
                capture_output=True,
                check=True,
                timeout=10,
            )
            payload = json.loads(result.stdout)
            self.assertIsInstance(payload, dict)
            self.assertEqual(payload["protocol"], "kernelyra-native-probe/1")
            self.assertEqual(payload["engine"], "native-cpp")
            routed = FormatRouter(executable).inspect(sample)
            self.assertEqual(routed["engine"], "native-cpp")
            self.assertEqual(routed["format"], "csv")
            with patch.dict("os.environ", {"KERNELYRA_NATIVE_PROBE": str(executable)}):
                workspace = Workspace.open(temporary / "installed-style-workspace")
                through_workspace = workspace.datasets.inspect(sample)
                workspace.close()
            self.assertEqual(through_workspace["engine"], "native-cpp")


if __name__ == "__main__":
    unittest.main()
