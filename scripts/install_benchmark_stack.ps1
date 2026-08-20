param(
    [string]$Python = "python",
    [string]$Environment = ".benchmarks\frameworks-venv"
)

$ErrorActionPreference = "Stop"
$environmentPath = [System.IO.Path]::GetFullPath($Environment)
& $Python -m venv $environmentPath
$venvPython = Join-Path $environmentPath "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install --only-binary=:all: ".[benchmark]"
& $venvPython scripts\benchmark_tabular_frameworks.py --output .benchmarks\framework-matrix.json
