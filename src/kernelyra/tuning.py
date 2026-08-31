"""Deterministic startup tuning for explicit execution and resource limits."""

from __future__ import annotations

from typing import Any

from .hardware import execution_policy, resolve_execution_target


def autotune_execution(
    hardware: dict[str, Any],
    *,
    records: int,
    features: int,
    batch_size: int,
    streaming: bool,
    execution_target: str = "auto",
    threads: int | None = None,
    cpu_percent: int = 100,
    memory_budget_bytes: int | None = None,
) -> dict[str, Any]:
    """Resolve a conservative, explainable execution plan before worker startup.

    This is intentionally a zero-cost tuner: it never runs a hidden benchmark
    or changes model quality.  Runtime telemetry exposes the selected values so
    a future adaptive pass can be evaluated against real throughput data.
    """
    execution = resolve_execution_target(execution_target, hardware)
    policy = execution_policy(hardware, execution_target=execution)
    cpu_threads = max(1, int(hardware.get("cpu_threads") or 1))
    native_threads = threads if threads is not None else round(
        cpu_threads * (max(10, min(100, int(cpu_percent))) / 100) * float(policy["native_thread_fraction"])
    )
    native_threads = max(1, min(cpu_threads, int(native_threads)))
    batch_bytes = max(1, int(batch_size)) * max(1, int(features) + 1) * 4
    requested_memory = int(memory_budget_bytes or 0)
    if requested_memory <= 0:
        requested_memory = int(float(hardware.get("ram_gb") or 8) * 1024**3)
    # The native arena owns only reusable feature/target batches. Reserving a
    # fixed 24-96 MiB for tiny batches wastes an explicit user budget. Two
    # aligned buffers cover x/y without an implicit allocator growth path.
    required_arena = ((batch_bytes * 2 + 63) // 64) * 64
    policy_cap = int(policy["arena_bytes"])
    memory_cap = max(64 * 1024, requested_memory // 16)
    arena_cap = min(policy_cap, memory_cap)
    arena_bytes = min(arena_cap, max(64 * 1024, required_arena))
    return {
        "mode": "automatic",
        "execution": execution,
        "native_threads": native_threads,
        "bulk_step_cap": int(policy["bulk_step_cap"]),
        "arena_bytes": arena_bytes,
        "arena_cap_bytes": arena_cap,
        "arena_required_bytes": required_arena,
        "arena_fits_batch": required_arena <= arena_cap,
        "memory_budget_bytes": requested_memory,
        "data_workers": int(policy["data_workers"]),
        "prefetch": int(policy["prefetch"]),
        "chunk_target_records": int(policy["chunk_target_records"]),
        "streaming": bool(streaming),
        "records": max(0, int(records)),
        "features": max(1, int(features)),
        "batch_bytes": batch_bytes,
        "strategy": str(policy["strategy"]),
    }
