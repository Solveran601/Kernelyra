from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SDKContractTests(unittest.TestCase):
    def test_real_sdk_smoke_runner_covers_all_primary_languages(self) -> None:
        source = (ROOT / "scripts/smoke_sdks.py").read_text(encoding="utf-8")
        for language in ("python", "go", "cpp", "rust", "csharp"):
            self.assertIn(f'"{language}"', source)

    def test_primary_sdks_expose_easy_api_and_protocol(self) -> None:
        expectations = {
            "sdks/go/kernelyra.go": ("type Config struct", "func (c *Client) Fit", "kernelyra-jsonl/1"),
            "sdks/rust/src/lib.rs": ("pub struct Config", "pub fn fit", "kernelyra-jsonl/1"),
            "sdks/cpp/kernelyra_client.hpp": ("class Config", "TrainingResult fit", 'call("train"'),
            "sdks/csharp/KernelyraClient.cs": ("sealed class Config", "TrainingResult Fit", "kernelyra-jsonl/1"),
        }
        for relative, markers in expectations.items():
            with self.subTest(relative=relative):
                source = (ROOT / relative).read_text(encoding="utf-8")
                for marker in markers:
                    self.assertIn(marker, source)

    def test_sdk_option_names_match_python_engine_contract(self) -> None:
        engine = (ROOT / "src/kernelyra/auto.py").read_text(encoding="utf-8")
        match = re.search(r"_DEFAULTS: dict\[str, Any\] = \{(?P<body>.*?)\n\}", engine, re.DOTALL)
        self.assertIsNotNone(match)
        names = set(re.findall(r'"([a-z_]+)"\s*:', match.group("body") if match else ""))
        required = {
            "target",
            "task",
            "backend",
            "architecture",
            "model_format",
            "profile",
            "batch_size",
            "max_steps",
            "target_metric",
            "cpu",
            "ram",
            "gpu",
            "seed",
            "learning_rate",
            "weight_decay",
            "hidden_layers",
            "precision",
            "data_workers",
            "prefetch",
            "evaluation_interval",
            "min_improvement",
            "degradation_margin",
            "degradation_patience",
            "early_stopping_patience",
            "target_patience",
        }
        self.assertEqual(required, names)
        for relative in (
            "src/kernelyra/easy.py",
            "sdks/go/kernelyra.go",
            "sdks/rust/src/lib.rs",
            "sdks/cpp/kernelyra_client.hpp",
            "sdks/csharp/KernelyraClient.cs",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            for name in required:
                self.assertIn(name, source, f"{name} is missing from {relative}")

    @unittest.skipUnless(shutil.which("g++"), "g++ is not installed")
    def test_cpp_headers_compile(self) -> None:
        output = ROOT / ".test_workspaces" / "sdk-cpp-object.o"
        output.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-pedantic",
                "-c",
                str(ROOT / "sdks/cpp/example.cpp"),
                "-o",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
