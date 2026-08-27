[CmdletBinding()]
param(
    [string]$Workspace = (Join-Path (Get-Location) ".kernelyra-user-journeys"),
    [string]$Python
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$pythonPath = if ($Python) { $Python } elseif ($env:KERNELYRA_PYTHON) { $env:KERNELYRA_PYTHON } else { Join-Path $repoRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python with Kernelyra was not found: $pythonPath. Create .venv and run '.\.venv\Scripts\python -m pip install -e .', or pass -Python."
}

$env:KERNELYRA_PYTHON = $pythonPath
Import-Module (Join-Path $repoRoot "powershell\Kernelyra.psd1") -Force

$null = New-Item -ItemType Directory -Force -Path $Workspace
$version = Invoke-Kernelyra -Workspace $Workspace -Arguments @("version")
$doctor = Invoke-Kernelyra -Workspace $Workspace -Arguments @("doctor")
$execution = Get-KernelyraExecution -Workspace $Workspace
$native = Get-KernelyraNativeStatus -Workspace $Workspace
$capabilities = Invoke-Kernelyra -Workspace $Workspace -Arguments @("capabilities")
$doctorSummary = [pscustomobject]@{
    ok = $doctor.ok
    version = $doctor.version
    checks = $doctor.checks
    default_execution = $doctor.default_execution
    hardware = $doctor.hardware
    optional_dependencies = $doctor.optional_dependencies
    warnings = $doctor.warnings
}
$capabilitySummary = [pscustomobject]@{
    contract_version = $capabilities.contract_version
    task_types = $capabilities.task_types
    format_counts = $capabilities.format_counts
    backends = @($capabilities.backends | Select-Object name, available, version, task_types, export_formats, diagnostic)
    architectures = @($capabilities.architectures | Select-Object id, implemented, modalities, tasks, backends, note)
    model_formats = @($capabilities.model_formats | Select-Object id, extensions, training_output, fine_tune, architectures, note)
}

[pscustomobject]@{
    version = $version
    doctor = $doctorSummary
    execution = $execution
    native = $native
    capabilities = $capabilitySummary
} | ConvertTo-Json -Depth 100

Write-Host "`nUX check: were the Python path, workspace, available backends and execution modes obvious from this output?"
