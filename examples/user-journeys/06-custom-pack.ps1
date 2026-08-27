[CmdletBinding()]
param(
    [ValidatePattern("^[a-z0-9][a-z0-9._-]{0,63}$")] [string]$Name = "ux-careful",
    [ValidateSet("careful", "balanced", "throughput", "maximum")] [string]$Base = "careful",
    [ValidateSet("bulk_training_dispatch", "thread_parallel_gradient", "parallel_data_prefetch", "wide_context_chunks", "expanded_tensor_arena")]
    [string]$AddAlgorithm = "thread_parallel_gradient",
    [switch]$Apply,
    [string]$Workspace = (Join-Path (Get-Location) ".kernelyra-user-journeys"),
    [string]$Python
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$pythonPath = if ($Python) { $Python } elseif ($env:KERNELYRA_PYTHON) { $env:KERNELYRA_PYTHON } else { Join-Path $repoRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw "Kernelyra Python was not found: $pythonPath" }
$env:KERNELYRA_PYTHON = $pythonPath
Import-Module (Join-Path $repoRoot "powershell\Kernelyra.psd1") -Force

$null = New-Item -ItemType Directory -Force -Path $Workspace
$builtIns = Get-KernelyraPack -Workspace $Workspace
$algorithms = Get-KernelyraPackAlgorithm -Workspace $Workspace
$tablePath = Get-KernelyraPackTablePath -Workspace $Workspace

if (-not $Apply) {
    [pscustomobject]@{
        mode = "preview"
        packs = $builtIns
        algorithms = $algorithms
        custom_pack_file = $tablePath
        next_command = ".\06-custom-pack.ps1 -Name $Name -Base $Base -AddAlgorithm $AddAlgorithm -Apply"
    } | ConvertTo-Json -Depth 100
    Write-Host "`nNo pack was created. Re-run with -Apply only after checking the preview."
    return
}

$created = Copy-KernelyraPack -Name $Name -Base $Base -Workspace $Workspace
$updated = Add-KernelyraPackAlgorithm -Name $Name -Algorithm $AddAlgorithm -Workspace $Workspace
$pack = Get-KernelyraPack -Name $Name -Workspace $Workspace

[pscustomobject]@{ created = $created; updated = $updated; effective_pack = $pack; custom_pack_file = $tablePath } | ConvertTo-Json -Depth 100
Write-Host "`nUX check: was it obvious that built-in packs stay immutable and only your clone was changed?"
