from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / ".test_workspaces"


@contextmanager
def isolated_workspace() -> Iterator[Path]:
    TEST_ROOT.mkdir(exist_ok=True)
    path = TEST_ROOT / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        resolved = path.resolve()
        if resolved.parent == TEST_ROOT.resolve() and resolved.exists():
            shutil.rmtree(resolved, ignore_errors=True)
        try:
            TEST_ROOT.rmdir()
        except OSError:
            pass
