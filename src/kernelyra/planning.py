"""Deterministic, context-safe planning helpers for streaming datasets.

The planner is deliberately independent of a dataframe library.  It can be
used before a dataset is materialised, so the same group (document, user,
conversation, or time-series entity) cannot leak between training and held-out
evaluation data.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, TypeVar

from .native_core import NativeCore, NativeCoreError

_SPLITS = ("train", "validation", "test")
_U64_MASK = (1 << 64) - 1
_ITEM = TypeVar("_ITEM")


@dataclass(frozen=True, slots=True)
class ContextChunk:
    """A half-open record range selected for a streaming read."""

    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "stop": self.stop, "size": self.size}


class ContextChunkPlanner:
    """Create variable chunks and deterministic context-preserving splits.

    The native Rust policy is used when the bundled core is available.  A
    bit-for-bit compatible Python policy keeps plans reproducible on systems
    where the native binary has not yet been installed.
    """

    def __init__(
        self,
        *,
        target_records: int = 4_096,
        minimum_records: int | None = None,
        maximum_records: int | None = None,
        validation_percent: int = 15,
        test_percent: int = 15,
        seed: int = 42,
        core: NativeCore | None = None,
    ) -> None:
        target_records = int(target_records)
        minimum_records = int(minimum_records if minimum_records is not None else target_records * 3 // 4)
        maximum_records = int(maximum_records if maximum_records is not None else target_records * 5 // 4)
        if target_records <= 0 or minimum_records <= 0 or maximum_records < minimum_records:
            raise ValueError("chunk limits must be positive and maximum_records must be at least minimum_records")
        if not minimum_records <= target_records <= maximum_records:
            raise ValueError("target_records must be between minimum_records and maximum_records")
        if not 0 <= validation_percent <= 95 or not 0 <= test_percent <= 95:
            raise ValueError("validation_percent and test_percent must be between 0 and 95")
        if validation_percent + test_percent > 95:
            raise ValueError("validation_percent + test_percent must leave at least 5% for training")
        self.target_records = target_records
        self.minimum_records = minimum_records
        self.maximum_records = maximum_records
        self.validation_percent = int(validation_percent)
        self.test_percent = int(test_percent)
        self.seed = int(seed) & _U64_MASK
        self._core = core
        self._native_checked = core is not None

    @staticmethod
    def _mix_u64(value: int) -> int:
        """SplitMix64 finalizer matching the native Rust context policy."""
        value = (value + 0x9E3779B97F4A7C15) & _U64_MASK
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _U64_MASK
        return (value ^ (value >> 31)) & _U64_MASK

    @staticmethod
    def _context_u64(context: Hashable) -> int:
        if isinstance(context, int):
            return context & _U64_MASK
        if isinstance(context, bytes):
            payload = context
        elif isinstance(context, str):
            payload = context.encode("utf-8")
        else:
            # Stable identifiers should normally be str/int.  Tagging the
            # fallback with its type prevents accidental collisions with text.
            payload = f"{type(context).__module__}.{type(context).__qualname__}:{context!r}".encode("utf-8")
        return int.from_bytes(blake2b(payload, digest_size=8, person=b"Kernelyra").digest(), "little")

    def _native_core(self) -> NativeCore | None:
        if not self._native_checked:
            self._native_checked = True
            try:
                self._core = NativeCore()
            except NativeCoreError:
                self._core = None
        return self._core

    @property
    def native_policy_active(self) -> bool:
        """Whether this planner is backed by the optional Rust policy binary."""
        return self._native_core() is not None

    def split_for(self, context: Hashable) -> str:
        """Return a stable split name for one context identifier."""
        key = self._context_u64(context)
        core = self._native_core()
        if core is not None:
            split_id = core.split_for_context(key, self.validation_percent, self.test_percent)
        else:
            bucket = self._mix_u64(key) % 100
            split_id = 1 if bucket < self.validation_percent else 2 if bucket < self.validation_percent + self.test_percent else 0
        return _SPLITS[split_id]

    def split_indices(self, contexts: Iterable[Hashable]) -> dict[str, list[int]]:
        """Group input positions by split while preserving their original order."""
        result: dict[str, list[int]] = {name: [] for name in _SPLITS}
        for index, context in enumerate(contexts):
            result[self.split_for(context)].append(index)
        return result

    def partition(self, items: Iterable[tuple[Hashable, _ITEM]]) -> dict[str, list[_ITEM]]:
        """Split ``(context, item)`` pairs without separating items in one context."""
        result: dict[str, list[_ITEM]] = {name: [] for name in _SPLITS}
        for context, item in items:
            result[self.split_for(context)].append(item)
        return result

    def _fallback_chunk_size(self, remaining_records: int, sequence: int) -> int:
        lower = min(max(self.target_records * 3 // 4, self.minimum_records), self.maximum_records)
        upper = min(max(self.target_records * 5 // 4, lower), self.maximum_records)
        proposed = lower + self._mix_u64(self.seed ^ sequence) % (upper - lower + 1)
        return max(min(proposed, remaining_records), min(remaining_records, self.minimum_records))

    def chunk_ranges(self, total_records: int) -> tuple[ContextChunk, ...]:
        """Return non-uniform, contiguous ranges that exactly cover ``total_records``."""
        remaining_records = int(total_records)
        if remaining_records < 0:
            raise ValueError("total_records cannot be negative")
        ranges: list[ContextChunk] = []
        start = 0
        sequence = 0
        core = self._native_core()
        while remaining_records:
            if core is None:
                size = self._fallback_chunk_size(remaining_records, sequence)
            else:
                size = core.next_chunk_size(
                    remaining_records,
                    self.target_records,
                    self.minimum_records,
                    self.maximum_records,
                    sequence,
                    self.seed,
                )
            if size <= 0 or size > remaining_records:
                raise RuntimeError("context chunk policy returned an invalid size")
            stop = start + size
            ranges.append(ContextChunk(start, stop))
            remaining_records -= size
            start = stop
            sequence += 1
        return tuple(ranges)

    def summary(self, total_records: int, *, preview: int = 12) -> dict[str, Any]:
        """Return JSON-ready planning diagnostics without exposing every range."""
        ranges = self.chunk_ranges(total_records)
        sizes = [item.size for item in ranges]
        visible = max(0, int(preview))
        return {
            "records": int(total_records),
            "chunk_count": len(ranges),
            "target_records": self.target_records,
            "minimum_records": min(sizes, default=0),
            "maximum_records": max(sizes, default=0),
            "native_policy_active": self.native_policy_active,
            "preview": [item.to_dict() for item in ranges[:visible]],
            "preview_truncated": len(ranges) > visible,
        }
