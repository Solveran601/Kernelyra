"""Print the actual loaded native-core capabilities as JSON.

This is useful before running a lab: it distinguishes the shipped v5 DLL from
the local experimental DLL without guessing from the Python package version.
"""

from __future__ import annotations

import json
import sys

from kernelyra.native_core import native_core_status


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    print(json.dumps(native_core_status(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
