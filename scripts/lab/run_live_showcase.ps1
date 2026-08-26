param(
    [string]$Python = "python",
    [string]$BuildDirectory = "native/build",
    [int]$Steps = 80
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONPATH = (Resolve-Path "src")
$env:KERNELYRA_NATIVE_CORE = (Resolve-Path (Join-Path $BuildDirectory "kernelyra_core.dll"))
& $Python scripts/lab/live_v5_showcase.py --steps $Steps
