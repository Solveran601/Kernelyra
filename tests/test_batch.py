from __future__ import annotations

import unittest

from kernelyra.batch import plan_batch


class BatchPlannerTests(unittest.TestCase):
    def test_auto_uses_dataset_shape(self) -> None:
        plan = plan_batch(records=2400, features=4, profile="eco", ram_percent=35, ram_gb=8)
        self.assertEqual(plan.applied, 32)
        self.assertEqual((plan.safe_min, plan.safe_max), (16, 64))
        self.assertEqual(plan.risk, "safe")

    def test_high_manual_batch_requires_confirmation(self) -> None:
        plan = plan_batch(records=2400, features=4, profile="eco", ram_percent=35, ram_gb=8, mode="manual", requested=512)
        self.assertEqual(plan.risk, "high")
        self.assertTrue(plan.requires_confirmation)

    def test_wide_dataset_reduces_recommendation(self) -> None:
        narrow = plan_batch(records=20_000, features=16, profile="balanced", ram_percent=55, ram_gb=16)
        wide = plan_batch(records=20_000, features=5000, profile="balanced", ram_percent=55, ram_gb=16)
        self.assertLess(wide.recommended, narrow.recommended)

    def test_workstation_profile_has_a_larger_safe_ceiling(self) -> None:
        performance = plan_batch(
            records=1_000_000,
            features=32,
            profile="performance",
            ram_percent=75,
            ram_gb=128,
        )
        workstation = plan_batch(
            records=1_000_000,
            features=32,
            profile="workstation",
            ram_percent=95,
            ram_gb=128,
        )
        self.assertGreater(workstation.safe_max, performance.safe_max)
        self.assertGreaterEqual(workstation.safe_max, 8192)


if __name__ == "__main__":
    unittest.main()
