from __future__ import annotations

import math
import unittest

from kernelyra.quality import QualityGate
from kernelyra.trace import TRACE_LIMIT, TrainingTrace


class TrainingSafetyTests(unittest.TestCase):
    def test_quality_gate_rejects_non_finite_metrics_before_checkpoint_promotion(self) -> None:
        gate = QualityGate(degradation_margin=.02, baseline_score=.91)
        result = gate.inspect(
            score=math.nan,
            loss=.2,
            metrics={"accuracy": .8, "nested": {"loss": math.inf}},
            best_score=.92,
        )
        self.assertFalse(result["finite"])
        self.assertEqual(result["status"], "invalid")
        self.assertIsNone(result["score"])
        self.assertIn("quality.score", result["invalid_paths"])
        self.assertIn("quality.validation.nested.loss", result["invalid_paths"])

    def test_trace_is_bounded_and_summarises_latest_evaluation(self) -> None:
        trace = TrainingTrace(limit=3)
        trace.add("started")
        trace.add("evaluation", step=10, score=.8, updates_per_second=12.5)
        trace.add("progress", step=11)
        trace.add("evaluation", step=20, score=.9, updates_per_second=15.0)
        self.assertEqual(len(trace.events), 3)
        self.assertEqual(trace.summary()["latest_step"], 20)
        self.assertEqual(trace.summary()["latest_score"], .9)
        self.assertLessEqual(len(TrainingTrace().events), TRACE_LIMIT)


if __name__ == "__main__":
    unittest.main()
