from __future__ import annotations

import argparse
import sys
from collections.abc import Collection
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True

FORBIDDEN_PREFIXES = {
    ".kernelyra",
    ".trainflow",
    "native/build",
    "native/rust/target",
    "native/core/rust/target",
    "sdks/csharp/bin",
    "sdks/csharp/obj",
    "sdks/rust/target",
}
FORBIDDEN_DIR_NAMES = {"arrays", "checkpoints", "uploads"}
FORBIDDEN_SUFFIXES = {".secret", ".sqlite3", ".sqlite3-wal", ".sqlite3-shm", ".pid", ".npz", ".pyc", ".pyo", ".pdb", ".exe", ".log"}
ALLOWED_PLATFORM_BINARIES = {"src/kernelyra/native_bin/kernelyra_core.dll"}
PERSONAL_PATH_MARKERS = ("c:" + "/users/" + "jamal", "c:" + "\\users\\" + "jamal")


def forbidden_name(raw_name: str, *, include_python_cache: bool = True) -> bool:
    name = raw_name.replace("\\", "/").strip("./")
    parts = tuple(part for part in name.split("/") if part)
    padded = "/" + name
    if any(padded.endswith("/" + prefix) or ("/" + prefix + "/") in padded for prefix in FORBIDDEN_PREFIXES):
        return True
    if any(part in FORBIDDEN_DIR_NAMES or part.lower().endswith(".egg-info") or part.startswith("pytest-cache-files-") for part in parts):
        return True
    if include_python_cache and "__pycache__" in parts:
        return True
    lower = name.lower()
    if lower in ALLOWED_PLATFORM_BINARIES:
        return False
    suffixes = FORBIDDEN_SUFFIXES if include_python_cache else FORBIDDEN_SUFFIXES - {".pyc", ".pyo"}
    if any(lower.endswith(suffix) for suffix in suffixes):
        return True
    return (lower.endswith(".lock") and (not parts or parts[-1].lower() != "cargo.lock")) or lower.endswith("demo_dataset.csv")


def find_forbidden(
    root: Path,
    *,
    include_python_cache: bool = True,
    ignore_top_level: Collection[str] = (),
    allow_generated_egg_info: bool = False,
) -> list[str]:
    ignored_roots = {".git", ".venv", "dist", "build", *ignore_top_level}
    found: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if allow_generated_egg_info and (
            relative == "src/kernelyra_ai.egg-info"
            or relative.startswith("src/kernelyra_ai.egg-info/")
        ):
            continue
        if relative.split("/", 1)[0] not in ignored_roots and forbidden_name(relative, include_python_cache=include_python_cache):
            found.append(relative)
    return sorted(set(found))


def find_personal_paths(root: Path, *, ignore_top_level: Collection[str] = ()) -> list[str]:
    ignored_roots = {".git", ".venv", "dist", "build", *ignore_top_level}
    text_suffixes = {".bat", ".cfg", ".ini", ".md", ".ps1", ".py", ".toml", ".txt", ".yml", ".yaml"}
    found: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative.split("/", 1)[0] in ignored_roots or path.suffix.lower() not in text_suffixes:
            continue
        try:
            value = path.read_text(encoding="utf-8").lower()
        except (OSError, UnicodeDecodeError):
            continue
        if any(marker in value for marker in PERSONAL_PATH_MARKERS):
            found.append(relative)
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-generated-egg-info",
        action="store_true",
        help="ignore the local egg-info directory created by an editable install",
    )
    args = parser.parse_args()
    local = {".kernelyra", ".trainflow", ".test_workspaces", ".benchmarks"}
    forbidden = find_forbidden(
        ROOT,
        ignore_top_level=local,
        allow_generated_egg_info=args.allow_generated_egg_info,
    )
    if forbidden:
        print("Forbidden source artifacts:")
        print("\n".join(forbidden))
        return 1
    personal = find_personal_paths(ROOT, ignore_top_level=local)
    if personal:
        print("Personal absolute paths:")
        print("\n".join(personal))
        return 1
    print("Clean source check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
