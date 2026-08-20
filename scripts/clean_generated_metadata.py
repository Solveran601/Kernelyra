from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIRECTORIES = (
    ROOT / "src" / "kernelyra_ai.egg-info",
    ROOT / "sdks" / "csharp" / "bin",
    ROOT / "sdks" / "csharp" / "obj",
    ROOT / "sdks" / "rust" / "target",
)
GENERATED_PARENTS = {target.parent.resolve() for target in GENERATED_DIRECTORIES}


def main() -> int:
    removed: list[str] = []
    for target in GENERATED_DIRECTORIES:
        if not target.exists() and not target.is_symlink():
            continue
        if target.parent.resolve() not in GENERATED_PARENTS:
            raise SystemExit(f"Refusing to remove unexpected path: {target}")
        relative = target.relative_to(ROOT).as_posix()
        if target.is_symlink():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        else:
            raise SystemExit(f"Refusing to remove unexpected non-directory: {target}")
        removed.append(relative)
    for target in sorted(ROOT.rglob("__pycache__"), key=lambda path: len(path.parts), reverse=True):
        resolved = target.resolve()
        if target.name != "__pycache__" or ROOT.resolve() not in resolved.parents:
            raise SystemExit(f"Refusing to remove unexpected Python cache: {resolved}")
        relative = target.relative_to(ROOT).as_posix()
        shutil.rmtree(target)
        removed.append(relative)
    print("Generated metadata cleanup: " + (", ".join(removed) if removed else "nothing to remove"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
