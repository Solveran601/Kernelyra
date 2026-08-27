"""Create and print a v5 native training plan without starting training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kernelyra import Config, Engine


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--workspace", type=Path, default=Path(".test_workspaces") / "plan-only")
    parser.add_argument("--cpu", type=int, default=50)
    parser.add_argument("--ram", type=int, default=40)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--pack", choices=("careful", "balanced", "throughput", "maximum"), default="careful")
    args = parser.parse_args()
    settings = (
        Config()
        .target(args.target)
        .backend("native")
        .cpu_only()
        .pack(args.pack)
        .resources(cpu=args.cpu, ram=args.ram, threads=args.threads)
    )
    with Engine(args.workspace.resolve()) as engine:
        plan = engine.plan(args.dataset.resolve(), settings=settings)
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
