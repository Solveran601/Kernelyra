from __future__ import annotations

import unittest
from pathlib import Path

from scripts.build_source_bundle import source_files
from scripts.verify_release import forbidden_archive_entry
from tests.clean_source import find_forbidden, find_personal_paths
from tests.helpers import ROOT, isolated_workspace


class CleanSourceTests(unittest.TestCase):
    def test_source_bundle_includes_native_and_sdk_build_metadata(self) -> None:
        names = {path.relative_to(ROOT).as_posix() for path in source_files()}
        for required in (
            "setup.py",
            "native/include/kernelyra_core.h",
            "native/bridge/cpp/core_abi.cpp",
            "src/kernelyra/formats.py",
            "src/kernelyra/architectures.py",
            "sdks/go/go.mod",
            "sdks/cpp/CMakeLists.txt",
            "sdks/csharp/Kernelyra.Client.csproj",
            "sdks/csharp/Directory.Build.props",
            "sdks/rust/Cargo.toml",
        ):
            self.assertIn(required, names)
        self.assertFalse(
            any(
                name.startswith("src/kernelyra/native_bin/")
                and Path(name).suffix in {".dll", ".so", ".dylib"}
                for name in names
            )
        )

    def test_repository_has_no_runtime_native_or_private_data(self) -> None:
        forbidden = find_forbidden(
            ROOT, ignore_top_level={".kernelyra", ".trainflow", ".test_workspaces", ".benchmarks"}
        )
        self.assertEqual(forbidden, [], "Forbidden release artifacts:\n" + "\n".join(forbidden))

    def test_repository_has_no_personal_absolute_paths(self) -> None:
        found = find_personal_paths(
            ROOT, ignore_top_level={".kernelyra", ".trainflow", ".test_workspaces", ".benchmarks"}
        )
        self.assertEqual(found, [], "Personal absolute paths:\n" + "\n".join(found))

    def test_scanner_rejects_runtime_secrets_and_python_cache(self) -> None:
        with isolated_workspace() as temporary:
            (temporary / ".kernelyra").mkdir()
            (temporary / ".kernelyra" / "agent.secret").write_text("leaked", encoding="ascii")
            cache = temporary / "package" / "__pycache__"
            cache.mkdir(parents=True)
            (cache / "module.pyc").write_bytes(b"cache")
            (temporary / "runtime.pid").write_text("42", encoding="ascii")
            forbidden = find_forbidden(temporary)
            self.assertIn(".kernelyra/agent.secret", forbidden)
            self.assertIn("package/__pycache__", forbidden)
            self.assertIn("package/__pycache__/module.pyc", forbidden)
            self.assertIn("runtime.pid", forbidden)

    def test_sdist_allows_only_standard_generated_metadata(self) -> None:
        sdist = Path("kernelyra_ai-0.3.0a1.tar.gz")
        source_zip = Path("kernelyra_ai-0.3.0a1-source.zip")
        metadata = "kernelyra_ai-0.3.0a1/src/kernelyra_ai.egg-info/SOURCES.txt"
        self.assertFalse(forbidden_archive_entry(sdist, metadata))
        self.assertTrue(forbidden_archive_entry(source_zip, metadata))
        self.assertTrue(
            forbidden_archive_entry(
                sdist,
                "kernelyra_ai-0.3.0a1/src/kernelyra_ai.egg-info/leaked.secret",
            )
        )


if __name__ == "__main__":
    unittest.main()
