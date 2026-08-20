"""Deterministic startup tuning for the four Kernelyra execution programs."""

from __future__ import annotations

from typing import Any

from .hardware import execution_policy, recommend_profile


def autotune_execution(
    profile: str,
    hardware: dict[str, Any],
    *,
    records: int,
    features: int,
    batch_size: int,
    streaming: bool,
) -> dict[str, Any]:
    """Resolve a conservative, explainable execution plan before worker startup.

    This is intentionally a zero-cost tuner: it never runs a hidden benchmark
    or changes model quality.  Runtime telemetry exposes the selected values so
    a future adaptive pass can be evaluated against real throughput data.
    """
    selected_profile = recommend_profile(hardware) if profile == "auto" else profile
    policy = execution_policy(selected_profile, hardware)
    cpu_threads = max(1, int(hardware.get("cpu_threads") or 1))
    native_threads = max(1, min(cpu_threads, round(cpu_threads * float(policy["native_thread_fraction"]))))
    batch_bytes = max(1, int(batch_size)) * max(1, int(features) + 1) * 4
    arena_cap = int(policy["arena_bytes"])
    # Reserve a profile-specific reusable working set.  The cap remains hard,
    # while one quarter makes the four modes materially different even for a
    # small first batch; larger batches grow the arena up to that cap.
    minimum_arena = max(8 * 1024**2, batch_bytes * 3)
    arena_bytes = min(arena_cap, max(minimum_arena, batch_bytes * 8, arena_cap // 4))
    return {
        "mode": policy["mode"],
        "profile": selected_profile,
        "native_threads": native_threads,
        "bulk_step_cap": int(policy["bulk_step_cap"]),
        "arena_bytes": arena_bytes,
        "arena_cap_bytes": arena_cap,
        "data_workers": int(policy["data_workers"]),
        "prefetch": int(policy["prefetch"]),
        "streaming": bool(streaming),
        "records": max(0, int(records)),
        "features": max(1, int(features)),
        "batch_bytes": batch_bytes,
        "strategy": str(policy["strategy"]),
    }
