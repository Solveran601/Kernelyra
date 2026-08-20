from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernelyra.native_core import NativeCore, build_native_core  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Kernelyra Windows Fortran/Zig native core")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = build_native_core(args.output)
    core = NativeCore(output)
    print(
        json.dumps(
            {
                "path": str(output),
                "version": core.version,
                "features": core.features,
                "components": core.components,
                "component_mask": core.component_mask,
                "enabled_component_mask": core.enabled_component_mask,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
