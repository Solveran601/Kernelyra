"""Build every primary SDK and perform a real two-step NumPy training run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from kernelyra import Config, Engine  # noqa: E402

LANGUAGES = ("python", "go", "cpp", "rust", "csharp")


def _dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("x1", "x2", "x3", "target"))
        for index in range(96):
            x1 = math.sin(index / 7)
            x2 = math.cos(index / 11)
            x3 = (index % 13) / 13
            writer.writerow((x1, x2, x3, int(x1 + x2 + x3 > .75)))


def _executable(explicit: str | None) -> str:
    if explicit:
        return str(Path(explicit).resolve())
    installed = shutil.which("kernelyra")
    if installed:
        return installed
    name = "kernelyra.exe" if sys.platform == "win32" else "kernelyra"
    binary_dir = "Scripts" if sys.platform == "win32" else "bin"
    candidate = Path(sys.executable).resolve().parent / binary_dir / name
    if candidate.is_file():
        return str(candidate)
    raise SystemExit("Kernelyra executable was not found; install the engine with: pip install -e .")


def _command(language: str, root: Path, dataset: Path, executable: str) -> tuple[list[str], Path]:
    workspace = root / language
    common = [str(dataset), "target", str(workspace), executable, "numpy", "2"]
    if language == "go":
        return ["go", "run", "./examples/easy", *common], ROOT / "sdks" / "go"
    if language == "rust":
        return [
            "cargo", "run", "--offline", "--target-dir", str(root / "rust-target"),
            "--example", "easy", "--", *common,
        ], ROOT / "sdks" / "rust"
    if language == "cpp":
        binary = root / ("kernelyra-cpp.exe" if sys.platform == "win32" else "kernelyra-cpp")
        compiler = shutil.which("g++") or "g++"
        build = subprocess.run(
            [
                compiler, "-std=c++17", "-Wall", "-Wextra", "-pedantic", "-Werror",
                str(ROOT / "sdks" / "cpp" / "example.cpp"), "-o", str(binary),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if build.returncode:
            raise RuntimeError(build.stdout + build.stderr)
        return [str(binary), *common], ROOT
    if language == "csharp":
        return [
            "dotnet", "run", "--project", str(ROOT / "sdks" / "csharp" / "examples"),
            "--configuration", "Release", "--", *common,
        ], ROOT
    raise ValueError(language)


def _python(root: Path, dataset: Path) -> dict[str, Any]:
    with Engine(root / "python") as engine:
        result = engine.fit(
            dataset,
            "target",
            settings=Config().backend("numpy").steps(2).goal(.95),
        )
    checkpoint = result.checkpoint
    return {
        "ok": bool(checkpoint and Path(checkpoint).is_file()),
        "status": result.run.status,
        "checkpoint": checkpoint,
    }


def _external(language: str, root: Path, dataset: Path, executable: str, timeout: float) -> dict[str, Any]:
    command, cwd = _command(language, root, dataset, executable)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
        env=environment,
    )
    output = (completed.stdout + completed.stderr).strip()
    checkpoint = output.rsplit("checkpoint=", 1)[-1].strip().splitlines()[0] if "checkpoint=" in output else ""
    return {
        "ok": completed.returncode == 0 and bool(checkpoint) and Path(checkpoint).is_file(),
        "returncode": completed.returncode,
        "checkpoint": checkpoint or None,
        "output": output[-2_000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=list(LANGUAGES))
    parser.add_argument("--executable")
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".test_workspaces" / "sdk-e2e")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.work_dir / "dataset.csv"
    _dataset(dataset)
    executable = _executable(args.executable)
    results: dict[str, dict[str, Any]] = {}
    for language in args.languages:
        tool = {"go": "go", "cpp": "g++", "rust": "cargo", "csharp": "dotnet"}.get(language)
        if tool and not shutil.which(tool):
            results[language] = {"ok": False, "error": f"{tool} is not installed"}
            continue
        try:
            results[language] = (
                _python(args.work_dir, dataset)
                if language == "python"
                else _external(language, args.work_dir, dataset, executable, args.timeout)
            )
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            results[language] = {"ok": False, "error": str(error)}
        print(f"{language}: {'OK' if results[language].get('ok') else 'FAILED'}")
    report = {
        "schema": "kernelyra-sdk-smoke/1",
        "executable": executable,
        "dataset": str(dataset),
        "results": results,
    }
    report_path = args.work_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Report: {report_path}")
    return 0 if all(result.get("ok") for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
