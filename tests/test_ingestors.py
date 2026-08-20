from __future__ import annotations

import unittest
from pathlib import Path

from kernelyra.ingestion.registry import IngestorRegistry


class IngestorTests(unittest.TestCase):
    def test_registry_is_closed_and_builtin_only(self) -> None:
        registry = IngestorRegistry()
        self.assertEqual(registry.names(), ["csv", "jsonl", "npz", "parquet"])
        self.assertFalse(hasattr(registry, "register"))
        self.assertFalse(hasattr(registry, "discover"))
        self.assertEqual(registry.for_path(Path("sample.csv")).name, "csv")
        self.assertIsNone(registry.for_path(Path("sample.toy")))


if __name__ == "__main__":
    unittest.main()
