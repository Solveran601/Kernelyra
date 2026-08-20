from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def test_windows_launchers_delegate_to_the_terminal_flow(self) -> None:
        batch = (ROOT / "start_worker.bat").read_text(encoding="utf-8")
        powershell = (ROOT / "start_worker.ps1").read_text(encoding="utf-8")
        for launcher in (batch, powershell):
            self.assertIn("worker.py", launcher)
            self.assertNotIn("daemon.secret", launcher)
        worker = (ROOT / "worker.py").read_text(encoding="utf-8")
        self.assertIn('arguments = sys.argv[1:] or ["--help"]', worker)
        self.assertNotIn('"desktop"', worker)
        self.assertIn("%*", batch)
        self.assertIn("@args", powershell)
        self.assertNotIn('"foreground"', worker)


if __name__ == "__main__":
    unittest.main()
