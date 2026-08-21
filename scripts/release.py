from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def run(*arguments: str) -> None:
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"}
    subprocess.run([sys.executable, "-B", *arguments], cwd=ROOT, env=environment, check=True)


def project() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def version() -> str:
    return str(project()["version"])


def check_versions() -> None:
    expected = version()
    values = {
        "src/kernelyra/__init__.py": f'__version__ = "{expected}"',
        "src/kernelyra/server.py": f'VERSION = "{expected}"',
        "src/kernelyra/cli.py": f'VERSION = "{expected}"',
    }
    mismatches = [name for name, marker in values.items() if marker not in (ROOT / name).read_text(encoding="utf-8")]
    if mismatches:
        raise SystemExit("Version mismatch: " + ", ".join(mismatches))


def check_release_tag() -> None:
    """Reject a GitHub release tag that would publish a different package version."""
    tag = os.environ.get("GITHUB_REF_NAME", "")
    if tag.startswith("v") and tag != f"v{version()}":
        raise SystemExit(f"Git tag {tag!r} does not match package version v{version()!s}")


def clean_build_state() -> None:
    for path in (ROOT / "build",):
        resolved = path.resolve()
        if resolved.parent != ROOT.resolve():
            raise SystemExit(f"Refusing cleanup outside project: {resolved}")
        shutil.rmtree(resolved, ignore_errors=True)
    if DIST.exists():
        for artifact in DIST.iterdir():
            if artifact.is_file():
                artifact.unlink()
            elif artifact.is_dir():
                shutil.rmtree(artifact)
    run("scripts/clean_generated_metadata.py")


def command_check() -> None:
    run("scripts/check_clean_source.py")
    check_versions()
    run("-m", "ruff", "check", ".")
    run("-m", "mypy", "src/kernelyra")
    run("scripts/check_clean_source.py")


def command_build() -> None:
    check_versions()
    check_release_tag()
    clean_build_state()
    run("scripts/build_native_core.py")
    run("-m", "build", "--wheel")
    run("-m", "build", "--sdist")
    run("scripts/clean_generated_metadata.py")
    run("scripts/build_source_bundle.py")
    run("scripts/clean_generated_metadata.py")
    run("scripts/check_clean_source.py")
    expected = version()
    platform_wheels = list(DIST.glob(f"kernelyra_ai-{expected}-py3-none-*.whl"))
    required = (
        DIST / f"kernelyra_ai-{expected}.tar.gz",
        DIST / f"kernelyra_ai-{expected}-source.zip",
    )
    missing = [path.name for path in required if not path.is_file()]
    if len(platform_wheels) != 1:
        missing.append(f"exactly one platform wheel (found {len(platform_wheels)})")
    if missing:
        raise SystemExit("Missing release artifacts: " + ", ".join(missing))


def command_verify() -> None:
    run("scripts/verify_release.py")
    archives = [str(path) for path in sorted(DIST.glob("*.whl")) + sorted(DIST.glob("*.tar.gz"))]
    if not archives:
        raise SystemExit("Build artifacts first")
    run("-m", "twine", "check", *archives)
    run("scripts/check_clean_source.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_bundle() -> None:
    artifacts = sorted(path for path in DIST.iterdir() if path.is_file() and path.suffix in {".whl", ".gz", ".zip"})
    if len(artifacts) != 3:
        raise SystemExit("Expected exactly wheel, sdist and source ZIP before bundle")
    hashes = {path.name: sha256(path) for path in artifacts}
    dependency_manifest = {
        "contract": "kernelyra-dependencies/1",
        "project": project()["name"],
        "version": version(),
        "python": project()["requires-python"],
        "dependencies": project()["dependencies"],
        "optional_dependencies": project().get("optional-dependencies", {}),
    }
    dependency_path = DIST / "DEPENDENCY_MANIFEST.json"
    dependency_path.write_text(json.dumps(dependency_manifest, indent=2, sort_keys=True), encoding="utf-8")
    current_version = version()
    manifest = {
        "contract": "kernelyra-release-manifest/1",
        "version": current_version,
        "status": "pre-release"
        if re.search(r"(?:a|b|rc|dev)\d", current_version, flags=re.IGNORECASE)
        else "stable",
        "artifacts": [
            {"name": path.name, "sha256": hashes[path.name], "size_bytes": path.stat().st_size}
            for path in artifacts
        ],
        "dependency_manifest": {"name": dependency_path.name, "sha256": sha256(dependency_path)},
        "published": False,
    }
    content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (ROOT / "RELEASE_MANIFEST.json").write_text(content, encoding="utf-8")
    (DIST / "RELEASE_MANIFEST.json").write_text(content, encoding="utf-8")
    sums = "".join(f"{value}  {name}\n" for name, value in sorted(hashes.items()))
    (ROOT / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    (DIST / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Kernelyra reproducible local release")
    parser.add_argument("command", choices=("check", "build", "verify", "bundle"))
    args = parser.parse_args()
    {"check": command_check, "build": command_build, "verify": command_verify, "bundle": command_bundle}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
