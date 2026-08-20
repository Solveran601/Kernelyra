from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

from kernelyra.cli import main


class CliConvenienceTests(unittest.TestCase):
    def test_modes_reports_all_four_programs(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["--json", "modes"]), 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(set(payload["modes"]), {"weak", "balanced", "performance", "workstation"})
        self.assertIn(payload["recommended_mode"], payload["modes"])

    def test_chunk_plan_is_json_and_never_emits_an_unbounded_range_list(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["--json", "chunk-plan", "100000", "--target-records", "100"]), 0)
        payload = json.loads(output.getvalue())
        self.assertGreater(payload["chunk_count"], len(payload["preview"]))
        self.assertTrue(payload["preview_truncated"])
        self.assertEqual(payload["preview"][0]["start"], 0)

    def test_tune_exposes_the_actual_worker_limits(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                main([
                    "--json", "tune", "--profile", "low-memory", "--records", "5000",
                    "--features", "12", "--batch-size", "16", "--streaming",
                ]),
                0,
            )
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["mode"], "weak")
        self.assertEqual(payload["bulk_step_cap"], 8)
        self.assertTrue(payload["streaming"])


if __name__ == "__main__":
    unittest.main()
