from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class BatchPlan:
    records: int
    features: int
    train_records: int
    mode: str
    recommended: int
    requested: int | None
    applied: int
    safe_min: int
    safe_max: int
    risk: str
    requires_confirmation: bool
    warnings: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _power_floor(value: int) -> int:
    value = max(2, int(value))
    return 1 << (value.bit_length() - 1)


def plan_batch(
    *,
    records: int,
    features: int,
    profile: str,
    ram_percent: int,
    ram_gb: float,
    mode: str = "auto",
    requested: int | None = None,
) -> BatchPlan:
    if mode not in {"auto", "manual"}:
        raise ConfigurationError("Batch mode must be auto or manual")
    if profile not in {"eco", "low-memory", "balanced", "performance", "workstation", "custom"}:
        raise ConfigurationError("Unknown hardware profile")
    validation_size = min(480, max(32, records // 5))
    train_records = max(2, records - validation_size)
    recommended = (
        4 if train_records < 64
        else 8 if train_records < 256
        else 16 if train_records < 1024
        else 32 if train_records < 10_000
        else 64 if train_records < 100_000
        else 128
    )
    if features > 4096:
        recommended = max(2, recommended // 4)
    elif features > 512:
        recommended = max(2, recommended // 2)

    profile_cap = {
        "eco": 64,
        "low-memory": 64,
        "balanced": 256,
        "performance": 512,
        # This is only a guard against pathological allocations, not a
        # workstation throttle. The real ceiling below is derived from RAM,
        # feature width and the training split.
        "workstation": 65_536,
        "custom": 256,
    }[profile]
    ram_percent = max(10, min(95, int(ram_percent)))
    total_bytes = max(1.0, ram_gb) * 1024**3
    reserve_bytes = max(2 * 1024**3, total_bytes * .05) if profile == "workstation" else 0
    requested_bytes = total_bytes * (ram_percent / 100)
    safety_factor = .80 if profile == "workstation" else .35
    usable_bytes = max(256 * 1024**2, min(requested_bytes, total_bytes - reserve_bytes) * safety_factor)
    bytes_per_sample = max(2048, features * 4 * 12)
    memory_cap = _power_floor(int(usable_bytes / bytes_per_sample))
    dataset_cap = _power_floor(max(2, train_records // 2))
    safe_max = max(2, min(profile_cap, memory_cap, dataset_cap))
    recommended = min(safe_max, _power_floor(recommended))
    safe_min = min(safe_max, max(2, recommended // 2))
    warnings: list[str] = []
    risk = "safe"
    applied = recommended
    requested_value: int | None = None
    if mode == "manual":
        if requested is None:
            raise ConfigurationError("Manual batch mode requires an integer batch size")
        requested_value = int(requested)
        if requested_value < 1:
            raise ConfigurationError("Batch size must be positive")
        hard_max = min(65_536 if profile == "workstation" else 4096, train_records)
        applied = max(1, min(requested_value, hard_max))
        if requested_value > hard_max:
            warnings.append(
                f"Requested {requested_value}; applied {applied} because the training split has {train_records} rows"
            )
        if applied > safe_max:
            risk = "high"
            warnings.extend(
                [
                    "A high batch can exhaust memory and may hide quality changes",
                    "OOM recovery uses the last checkpoint; automatic mode can reduce the batch safely",
                ]
            )
        elif applied < safe_min:
            risk = "caution"
            warnings.append("A small batch can increase gradient noise and reduce throughput")
        elif applied != recommended:
            risk = "caution"
            warnings.append("The requested value is safe but differs from the calculated optimum")
    else:
        warnings.append(
            "Automatic mode changes the batch only inside the safe range based on validation results"
        )
    reason = (
        f"{records} rows, {features} features, {train_records} training rows; "
        f"profile {profile} and RAM limit {ram_percent}%"
    )
    return BatchPlan(
        records,
        features,
        train_records,
        mode,
        recommended,
        requested_value,
        applied,
        safe_min,
        safe_max,
        risk,
        risk == "high",
        warnings,
        reason,
    )
