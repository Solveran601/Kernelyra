from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from tests.clean_source import find_forbidden, find_personal_paths  # noqa: E402


def main() -> int:
    local = {".kernelyra", ".trainflow", ".test_workspaces", ".benchmarks"}
    forbidden = find_forbidden(ROOT, ignore_top_level=local)
    if forbidden:
        print("Forbidden source artifacts:")
        print("\n".join(forbidden))
        return 1
    personal = find_personal_paths(ROOT, ignore_top_level=local)
    if personal:
        print("Personal absolute paths:")
        print("\n".join(personal))
        return 1
    print("Clean source check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
