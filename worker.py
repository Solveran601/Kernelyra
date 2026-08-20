"""Compatibility launcher for the terminal-first Kernelyra engine.

Importing this module has no runtime side effects. Prefer the ``kernelyra`` CLI
after installing the package.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    from kernelyra.cli import main as cli_main
    arguments = sys.argv[1:] or ["--help"]
    return cli_main(["--workspace", str(ROOT), *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
