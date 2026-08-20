from __future__ import annotations

import shutil
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.sdist import sdist


class PlatformNativeWheel(bdist_wheel):
    """Tag the ctypes native runtime as a Python-version-independent platform wheel."""

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def run(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Kernelyra 0.3 is released for Windows only")
        suffix = ".dll"
        native_bin = Path(__file__).parent / "src" / "kernelyra" / "native_bin"
        if not any(native_bin.glob(f"*{suffix}")):
            raise RuntimeError(
                f"Native runtime {suffix} is missing; run `python scripts/build_native_core.py` before building a wheel"
            )
        super().run()

    def get_tag(self) -> tuple[str, str, str]:
        _python, _abi, platform = super().get_tag()
        return "py3", "none", platform


class SourceOnlyDistribution(sdist):
    """Keep Windows binaries out of source distributions."""

    def get_file_list(self) -> None:
        super().get_file_list()
        marker = "src/kernelyra/native_bin/"
        self.filelist.files = [
            name
            for name in self.filelist.files
            if not name.replace("\\", "/").startswith(marker)
        ]

    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        super().make_release_tree(base_dir, files)
        native_bin = Path(base_dir) / "src" / "kernelyra" / "native_bin"
        if native_bin.is_dir():
            shutil.rmtree(native_bin)


setup(cmdclass={"bdist_wheel": PlatformNativeWheel, "sdist": SourceOnlyDistribution})
