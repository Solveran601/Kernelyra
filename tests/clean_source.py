from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

FORBIDDEN_PREFIXES = {
    ".kernelyra",
    ".trainflow",
    "native/build",
    "native/rust/target",
    "sdks/csharp/bin",
    "sdks/csharp/obj",
    "sdks/rust/target",
}
FORBIDDEN_DIR_NAMES = {"arrays", "checkpoints", "uploads"}
FORBIDDEN_SUFFIXES = {
    ".secret",
    ".sqlite3",
    ".sqlite3-wal",
    ".sqlite3-shm",
    ".pid",
    ".npz",
    ".pyc",
    ".pyo",
    ".pdb",
    ".exe",
    ".log",
}
ALLOWED_PLATFORM_BINARIES = {
    "src/kernelyra/native_bin/kernelyra_core.dll",
}
PERSONAL_PATH_MARKERS = ("c:" + "/users/" + "jamal", "c:" + "\\users\\" + "jamal")


def forbidden_name(raw_name: str, *, include_python_cache: bool = True) -> bool:
    name = raw_name.replace("\\", "/").strip("./")
    parts = tuple(part for part in name.split("/") if part)
    padded = "/" + name
    if any(
        padded.endswith("/" + prefix) or ("/" + prefix + "/") in padded
        for prefix in FORBIDDEN_PREFIXES
    ):
        return True
    if any(part in FORBIDDEN_DIR_NAMES for part in parts):
        return True
    if any(part.lower().endswith(".egg-info") for part in parts):
        return True
    if any(part.startswith("pytest-cache-files-") for part in parts):
        return True
    if include_python_cache and "__pycache__" in parts:
        return True
    lower = name.lower()
    if lower in ALLOWED_PLATFORM_BINARIES:
        return False
    suffixes = FORBIDDEN_SUFFIXES if include_python_cache else FORBIDDEN_SUFFIXES - {".pyc", ".pyo"}
    if any(lower.endswith(suffix) for suffix in suffixes):
        return True
    if lower.endswith(".lock") and (not parts or parts[-1].lower() != "cargo.lock"):
        return True
    return lower.endswith("demo_dataset.csv")


def find_forbidden(
    root: Path,
    *,
    include_python_cache: bool = True,
    ignore_top_level: Collection[str] = (),
) -> list[str]:
    ignored_roots = {".git", ".venv", "dist", "build", *ignore_top_level}
    found: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative.split("/", 1)[0] in ignored_roots:
            continue
        if forbidden_name(relative, include_python_cache=include_python_cache):
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
