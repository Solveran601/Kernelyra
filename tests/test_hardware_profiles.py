from __future__ import annotations

import unittest
from unittest.mock import patch

from kernelyra.hardware import (
    EXECUTION_MODES,
    PROFILE_PRESETS,
    detect_hardware,
    execution_policy,
    recommend_profile,
)
from kernelyra.runtime import TrainingRuntime
from kernelyra.tuning import autotune_execution


class HardwareProfileTests(unittest.TestCase):
    def test_four_hardware_levels_are_available(self) -> None:
        self.assertTrue(
            {"low-memory", "balanced", "performance", "workstation"}.issubset(PROFILE_PRESETS)
        )
        self.assertEqual(set(EXECUTION_MODES), {"weak", "balanced", "performance", "workstation"})

    def test_profiles_resolve_to_four_distinct_execution_programs(self) -> None:
        weak = execution_policy("low-memory", {"gpu_available": False})
        balanced = execution_policy("balanced", {"gpu_available": False})
        powerful = execution_policy("performance", {"gpu_available": True})
        workstation = execution_policy("workstation", {"gpu_available": True})
        self.assertEqual(
            [weak["mode"], balanced["mode"], powerful["mode"], workstation["mode"]],
            ["weak", "balanced", "performance", "workstation"],
        )
        self.assertEqual(weak["backend_order"][:2], ("native", "numpy"))
        self.assertEqual(powerful["backend_order"][:2], ("torch", "tensorflow"))
        self.assertLess(weak["prefetch"], workstation["prefetch"])

    def test_non_nvidia_accelerator_hint_enables_gpu_policy(self) -> None:
        with patch.dict("os.environ", {"KERNELYRA_ACCELERATOR": "directml"}, clear=False):
            hardware = detect_hardware()
        self.assertTrue(hardware["gpu_available"])
        self.assertIn({"kind": "directml", "source": "KERNELYRA_ACCELERATOR"}, hardware["accelerators"])

    def test_recommends_gpu_workstation(self) -> None:
        profile = recommend_profile(
            {
                "cpu_threads": 32,
                "ram_gb": 64,
                "nvidia_gpus": [{"vram_gb": 24}],
            }
        )
        self.assertEqual(profile, "workstation")

    def test_recommends_cpu_only_workstation_when_resources_are_exceptional(self) -> None:
        profile = recommend_profile(
            {"cpu_threads": 64, "ram_gb": 128, "nvidia_gpus": []}
        )
        self.assertEqual(profile, "workstation")

    def test_workstation_uses_uncapped_compute_with_an_os_memory_reserve(self) -> None:
        preset = PROFILE_PRESETS["workstation"]
        self.assertEqual((preset["cpu"], preset["gpu"]), (100, 100))
        self.assertEqual(preset["ram"], 95)
        self.assertGreaterEqual(preset["reserve_ram_gb"], 2)
        self.assertEqual(
            (TrainingRuntime.CPU_CAP, TrainingRuntime.RAM_CAP, TrainingRuntime.GPU_CAP),
            (100, 100, 100),
        )

    def test_autotuner_translates_the_four_programs_to_worker_limits(self) -> None:
        hardware = {"cpu_threads": 32, "ram_gb": 64, "gpu_available": False, "nvidia_gpus": []}
        plans = [
            autotune_execution(
                profile,
                hardware,
                records=500_000,
                features=128,
                batch_size=64,
                streaming=True,
            )
            for profile in ("low-memory", "balanced", "performance", "workstation")
        ]
        self.assertEqual([plan["mode"] for plan in plans], ["weak", "balanced", "performance", "workstation"])
        self.assertEqual([plan["native_threads"] for plan in plans], [8, 16, 24, 32])
        self.assertEqual([plan["bulk_step_cap"] for plan in plans], [8, 32, 100, 100])
        self.assertLess(plans[0]["arena_bytes"], plans[-1]["arena_bytes"])
        self.assertTrue(all(plan["streaming"] for plan in plans))


if __name__ == "__main__":
    unittest.main()
