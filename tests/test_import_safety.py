from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ImportSafetyTests(unittest.TestCase):
    def test_import_has_no_runtime_side_effects(self) -> None:
        code = """
import json, sys, threading
before = {thread.name for thread in threading.enumerate()}
import kernelyra
after = {thread.name for thread in threading.enumerate()}
print(json.dumps({'tensorflow': 'tensorflow' in sys.modules, 'threads': sorted(after-before)}))
"""
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, text=True, capture_output=True, check=True, timeout=30)
        payload = json.loads(result.stdout.strip())
        self.assertFalse(payload["tensorflow"])
        self.assertEqual(payload["threads"], [])


if __name__ == "__main__":
    unittest.main()
