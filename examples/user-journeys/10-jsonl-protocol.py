"""User journey: inspect the stable local JSONL protocol used by C/C++/Rust SDKs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def ask(process: subprocess.Popen[str], request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("Kernelyra RPC pipes are unavailable")
    process.stdin.write(json.dumps({"id": request_id, "method": method, "params": params}) + "\n")
    process.stdin.flush()
    response = process.stdout.readline()
    if not response:
        stderr = process.stderr.read() if process.stderr is not None else ""
        raise RuntimeError(f"Kernelyra RPC ended unexpectedly: {stderr}")
    return json.loads(response)


def capability_summary(capabilities: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": capabilities.get("contract_version"),
        "task_types": capabilities.get("task_types"),
        "format_counts": capabilities.get("format_counts"),
        "backends": [
            {key: backend.get(key) for key in ("name", "available", "version", "task_types", "diagnostic")}
            for backend in capabilities.get("backends", [])
        ],
        "architectures": [
            {key: architecture.get(key) for key in ("id", "implemented", "modalities", "tasks", "backends", "note")}
            for architecture in capabilities.get("architectures", [])
        ],
        "model_formats": [
            {key: model_format.get(key) for key in ("id", "extensions", "training_output", "fine_tune", "architectures", "note")}
            for model_format in capabilities.get("model_formats", [])
        ],
    }


def plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        key: plan.get(key)
        for key in (
            "dataset",
            "target",
            "task",
            "backend",
            "architecture",
            "model_format",
            "execution",
            "algorithm_pack",
            "batch_size",
            "records_estimate",
            "features_estimate",
            "data_mode",
            "sources",
            "warnings",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("target")
    parser.add_argument("--workspace", type=Path, default=Path(".kernelyra-user-journeys"))
    args = parser.parse_args()
    if not args.dataset.is_file():
        parser.error(f"dataset does not exist: {args.dataset}")

    command = [sys.executable, "-m", "kernelyra", "--workspace", str(args.workspace), "rpc"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if process.stdout is None:
            raise RuntimeError("Kernelyra RPC stdout is unavailable")
        ready = json.loads(process.stdout.readline())
        ping = ask(process, 1, "ping", {})
        capabilities = ask(process, 2, "capabilities", {})
        planned = ask(
            process,
            3,
            "plan",
            {
                "dataset": str(args.dataset.resolve()),
                "target": args.target,
                "execution": "cpu",
                "algorithm_pack": "balanced",
                "cpu": 80,
                "ram": 70,
                "threads": 4,
            },
        )
        if capabilities.get("ok"):
            capabilities["result"] = capability_summary(capabilities["result"])
        if planned.get("ok"):
            planned["result"] = plan_summary(planned["result"])
        responses = [ping, capabilities, planned]
        print(json.dumps({"command": command, "ready": ready, "responses": responses}, ensure_ascii=False, indent=2))
        print("\nUX check: could another language implement this request/response contract without guessing hidden Python objects?")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()
