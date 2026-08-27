"""Show the actual v5 route and bounded inspection for one tabular file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kernelyra.ingestion.router import FormatRouter
from kernelyra.streaming import build_stream_spec


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--stream-spec", action="store_true", help="Also build the streaming contract")
    args = parser.parse_args()
    path = args.dataset.expanduser().resolve()
    result: dict[str, object] = {"route": FormatRouter().inspect(path)}
    if args.stream_spec:
        result["stream_spec"] = build_stream_spec(path, target=args.target)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
