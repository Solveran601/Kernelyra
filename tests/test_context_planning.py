from __future__ import annotations

import unittest
from unittest.mock import patch

from kernelyra import Config, ContextChunkPlanner
from kernelyra.native_core import NativeCoreError


class ContextPlanningTests(unittest.TestCase):
    def test_variable_ranges_are_deterministic_and_cover_every_record(self) -> None:
        planner = ContextChunkPlanner(target_records=100, minimum_records=75, maximum_records=125, seed=19)
        first = planner.chunk_ranges(1_000)
        second = planner.chunk_ranges(1_000)
        self.assertEqual(first, second)
        self.assertEqual(first[0].start, 0)
        self.assertEqual(first[-1].stop, 1_000)
        self.assertEqual(sum(item.size for item in first), 1_000)
        self.assertGreater(len({item.size for item in first[:-1]}), 1)

    def test_a_context_never_leaks_between_splits(self) -> None:
        planner = ContextChunkPlanner(seed=7)
        indices = planner.split_indices(["document-a", "document-b", "document-a", "document-c"])
        assigned = {split: set(values) for split, values in indices.items()}
        self.assertTrue(any({0, 2}.issubset(values) for values in assigned.values()))
        partitioned = planner.partition([("document-a", "one"), ("document-a", "two"), ("document-b", "three")])
        self.assertTrue(any({"one", "two"}.issubset(values) for values in partitioned.values()))

    def test_python_fallback_matches_the_native_policy(self) -> None:
        native = ContextChunkPlanner(target_records=100, minimum_records=75, maximum_records=125, seed=19)
        expected_ranges = native.chunk_ranges(1_000)
        expected_splits = [native.split_for(value) for value in (7, "doc-a", "doc-b")]
        with patch("kernelyra.planning.NativeCore", side_effect=NativeCoreError("not installed")):
            fallback = ContextChunkPlanner(target_records=100, minimum_records=75, maximum_records=125, seed=19)
            self.assertFalse(fallback.native_policy_active)
            self.assertEqual(fallback.chunk_ranges(1_000), expected_ranges)
            self.assertEqual([fallback.split_for(value) for value in (7, "doc-a", "doc-b")], expected_splits)

    def test_summary_is_bounded_and_easy_modes_map_to_the_four_programs(self) -> None:
        planner = ContextChunkPlanner(target_records=10, minimum_records=8, maximum_records=12)
        summary = planner.summary(1_000, preview=3)
        self.assertEqual(len(summary["preview"]), 3)
        self.assertTrue(summary["preview_truncated"])
        self.assertEqual(Config().weak().to_dict()["profile"], "low-memory")
        self.assertEqual(Config().medium().to_dict()["profile"], "balanced")
        self.assertEqual(Config().powerful().to_dict()["profile"], "performance")

    def test_invalid_limits_are_rejected_before_planning(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_records"):
            ContextChunkPlanner(target_records=10, minimum_records=11, maximum_records=12)
        with self.assertRaisesRegex(ValueError, "leave at least"):
            ContextChunkPlanner(validation_percent=80, test_percent=16)


if __name__ == "__main__":
    unittest.main()
