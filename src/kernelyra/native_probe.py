from __future__ import annotations

import importlib
import importlib.resources
import os
import shutil
from pathlib import Path

from .errors import ConfigurationError

PROBE_NAMES = ("dataset_probe.exe", "dataset_probe") if os.name == "nt" else ("dataset_probe", "dataset_probe.exe")


def resolve_native_probe(workspace_root: str | Path) -> Path | None:
    """Resolve the optional native probe without depending on a source checkout."""
    configured = os.environ.get("KERNELYRA_NATIVE_PROBE")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not candidate.is_file():
            raise ConfigurationError(f"KERNELYRA_NATIVE_PROBE указывает на отсутствующий файл: {candidate}")
        return candidate

    for name in PROBE_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()

    packaged = _platform_package_probe()
    if packaged:
        return packaged

    root = Path(workspace_root).expanduser().resolve()
    for name in PROBE_NAMES:
        for relative in (Path("native") / "cpp" / "build" / name, Path("native") / "bin" / name):
            candidate = root / relative
            if candidate.is_file():
                return candidate.resolve()
    return None


def _platform_package_probe() -> Path | None:
    try:
        module = importlib.import_module("kernelyra_native")
    except ModuleNotFoundError:
        return None
    getter = getattr(module, "get_probe_path", None)
    if callable(getter):
        candidate = Path(getter()).expanduser().resolve()
        return candidate if candidate.is_file() else None
    try:
        resources = importlib.resources.files("kernelyra_native")
    except (ModuleNotFoundError, TypeError):
        return None
    for name in PROBE_NAMES:
        resource_candidate = resources.joinpath("bin", name)
        try:
            path = Path(str(resource_candidate)).resolve()
        except TypeError:
            continue
        if path.is_file():
            return path
    return None
