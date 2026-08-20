from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from tests.clean_source import find_forbidden, find_personal_paths, forbidden_name  # noqa: E402

ROOT_FILES = (
    ".gitignore",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "pyproject.toml",
    "README.md",
    "requirements.txt",
    "SECURITY.md",
    "setup.py",
    "start_worker.bat",
    "start_worker.ps1",
    "THIRD_PARTY_NOTICES.md",
    "kernelyra.example.toml",
    "worker.py",
)
SOURCE_DIRS = (
    ".github",
    "constraints",
    "docs",
    "examples",
    "native",
    "packages",
    "scripts",
    "sdks",
    "src",
    "tests",
)


def source_files() -> list[Path]:
    selected: list[Path] = []
    for name in ROOT_FILES:
        path = ROOT / name
        if not path.is_file():
            raise SystemExit(f"Required source file is missing: {name}")
        selected.append(path)
    for name in SOURCE_DIRS:
        directory = ROOT / name
        if not directory.is_dir():
            raise SystemExit(f"Required source directory is missing: {name}")
        selected.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and not (
                path.parent == ROOT / "src" / "kernelyra" / "native_bin"
                and path.suffix.lower() in {".dll", ".so", ".dylib"}
            )
        )

    unique = sorted(set(selected), key=lambda item: item.relative_to(ROOT).as_posix())
    for path in unique:
        relative = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            raise SystemExit(f"Source bundle refuses symbolic link: {relative}")
        if forbidden_name(relative):
            raise SystemExit(f"Forbidden source bundle entry: {relative}")
    return unique


def validate_tree() -> None:
    local = {".kernelyra", ".trainflow", ".test_workspaces", ".benchmarks"}
    forbidden = find_forbidden(ROOT, ignore_top_level=local)
    if forbidden:
        raise SystemExit("Forbidden source artifacts:\n" + "\n".join(forbidden))
    personal = find_personal_paths(ROOT, ignore_top_level=local)
    if personal:
        raise SystemExit("Personal absolute paths:\n" + "\n".join(personal))


def main() -> int:
    validate_tree()
    with (ROOT / "pyproject.toml").open("rb") as handle:
        version = str(tomllib.load(handle)["project"]["version"])
    bundle_root = f"kernelyra_ai-{version}"
    output = ROOT / "dist" / f"kernelyra_ai-{version}-source.zip"
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in source_files():
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(f"{bundle_root}/{relative}", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        forbidden_entries = [name for name in names if forbidden_name(name)]
        if forbidden_entries:
            raise SystemExit("Forbidden entries in source ZIP:\n" + "\n".join(forbidden_entries))
        temporary_base = ROOT / "build"
        temporary_base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="source-check-", dir=temporary_base) as temporary:
            extract_root = Path(temporary)
            archive.extractall(extract_root)
            unpacked = extract_root / bundle_root
            result = subprocess.run(
                [sys.executable, "-B", str(unpacked / "scripts" / "check_clean_source.py")],
                cwd=unpacked,
                text=True,
                encoding="utf-8",
                capture_output=True,
            )
            if result.returncode:
                raise SystemExit(
                    "Extracted source ZIP failed clean-source verification:\n"
                    + result.stdout
                    + result.stderr
                )

    print(f"Source bundle: {output}")
    print("Extracted source bundle check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
