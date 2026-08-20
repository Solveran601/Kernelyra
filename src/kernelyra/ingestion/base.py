from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class Ingestor(Protocol):
    name: str
    extensions: tuple[str, ...]

    def inspect(self, path: Path) -> dict[str, Any]: ...
    def import_file(self, path: Path, target: str | None = None) -> tuple[dict[str, Any], Any, Any]: ...
