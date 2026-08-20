from __future__ import annotations

import unittest
from pathlib import Path


class HeadlessTests(unittest.TestCase):
    def test_browser_assets_are_absent(self) -> None:
        root = Path(__file__).resolve().parents[1]
        browser_files = {
            path.resolve()
            for suffix in ("*.html", "*.css", "*.js", "*.jsx", "*.ts", "*.tsx")
            for path in (root / "src").rglob(suffix)
        }
        self.assertEqual(browser_files, set())
        self.assertFalse((root / "src" / "kernelyra" / "web" / "index.html").exists())

    def test_cli_has_no_desktop_command(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "src" / "kernelyra" / "cli.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('add_parser("desktop")', source)
        self.assertNotIn('add_parser("plugin")', source)


if __name__ == "__main__":
    unittest.main()
