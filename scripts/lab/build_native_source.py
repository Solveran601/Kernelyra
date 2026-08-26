"""Build a local native DLL from the checked-in Rust, Zig, Fortran, C and C++.

The output must be outside ``src/kernelyra/native_bin`` so this script cannot
overwrite the DLL that would be packaged for users.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kernelyra.native_core import build_native_core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".test_workspaces") / "manual-native")
    args = parser.parse_args()
    target = args.output.resolve()
    if "native_bin" in target.parts:
        raise SystemExit("Refusing to overwrite a packaged native_bin directory; choose a lab output directory.")
    print(build_native_core(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
