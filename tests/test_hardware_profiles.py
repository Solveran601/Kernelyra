from __future__ import annotations

import unittest

from kernelyra.hardware import PROFILE_PRESETS, recommend_profile
from kernelyra.runtime import TrainingRuntime


class HardwareProfileTests(unittest.TestCase):
    def test_four_hardware_levels_are_available(self) -> None:
        self.assertTrue(
            {"low-memory", "balanced", "performance", "workstation"}.issubset(PROFILE_PRESETS)
        )

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


if __name__ == "__main__":
    unittest.main()
