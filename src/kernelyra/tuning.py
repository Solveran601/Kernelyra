"""Deterministic startup tuning for explicit execution and algorithm packs."""

from __future__ import annotations

from typing import Any

from .hardware import execution_policy, legacy_profile_for_pack, resolve_algorithm_pack, resolve_execution_target


def autotune_execution(
    profile: str,
    hardware: dict[str, Any],
    *,
    records: int,
    features: int,
    batch_size: int,
    streaming: bool,
    execution_target: str = "auto",
    algorithm_pack: str | None = None,
    threads: int | None = None,
    cpu_percent: int = 100,
) -> dict[str, Any]:
    """Resolve a conservative, explainable execution plan before worker startup.

    This is intentionally a zero-cost tuner: it never runs a hidden benchmark
    or changes model quality.  Runtime telemetry exposes the selected values so
    a future adaptive pass can be evaluated against real throughput data.
    """
    pack = resolve_algorithm_pack(algorithm_pack or profile)
    selected_profile = legacy_profile_for_pack(pack)
    execution = resolve_execution_target(execution_target, hardware)
    policy = execution_policy(
        selected_profile, hardware, execution_target=execution, algorithm_pack=pack
    )
    algorithms = set(policy["algorithms"])
    cpu_threads = max(1, int(hardware.get("cpu_threads") or 1))
    native_threads = threads if threads is not None else round(
        cpu_threads * (max(10, min(100, int(cpu_percent))) / 100) * float(policy["native_thread_fraction"])
    )
    native_threads = max(1, min(cpu_threads, int(native_threads)))
    if "thread_parallel_gradient" not in algorithms:
        native_threads = 1
    batch_bytes = max(1, int(batch_size)) * max(1, int(features) + 1) * 4
    arena_cap = int(policy["arena_bytes"])
    if "expanded_tensor_arena" not in algorithms:
        arena_cap = min(arena_cap, 96 * 1024**2)
    # Reserve a profile-specific reusable working set.  The cap remains hard,
    # while one quarter makes the four modes materially different even for a
    # small first batch; larger batches grow the arena up to that cap.
    minimum_arena = max(8 * 1024**2, batch_bytes * 3)
    arena_bytes = min(arena_cap, max(minimum_arena, batch_bytes * 8, arena_cap // 4))
    return {
        "mode": pack,
        "profile": selected_profile,
        "execution": execution,
        "algorithm_pack": pack,
        "pack_base": str(policy["base"]),
        "pack_algorithms": sorted(algorithms),
        "native_threads": native_threads,
        "bulk_step_cap": int(policy["bulk_step_cap"]) if "bulk_training_dispatch" in algorithms else 1,
        "arena_bytes": arena_bytes,
        "arena_cap_bytes": arena_cap,
        "data_workers": int(policy["data_workers"]) if "parallel_data_prefetch" in algorithms else 0,
        "prefetch": int(policy["prefetch"]) if "parallel_data_prefetch" in algorithms else 1,
        "chunk_target_records": (
            int(policy["chunk_target_records"])
            if "wide_context_chunks" in algorithms
            else min(4096, int(policy["chunk_target_records"]))
        ),
        "streaming": bool(streaming),
        "records": max(0, int(records)),
        "features": max(1, int(features)),
        "batch_bytes": batch_bytes,
        "strategy": str(policy["strategy"]),
    }
