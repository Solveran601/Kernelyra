[CmdletBinding()]
param(
    [Parameter(Mandatory)] [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })] [string]$Dataset,
    [Parameter(Mandatory)] [string]$Target,
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
$doctor = Get-KernelyraDataContract -Path $Dataset -Target $Target -Workspace $Workspace
$plan = Get-KernelyraPlan -Dataset $Dataset -Target $Target -Workspace $Workspace
$doctorSummary = [pscustomobject]@{
    format = $doctor.inspection.format
    trainable = $doctor.inspection.trainable
    suggested_target = $doctor.inspection.suggested_target
    schema = $doctor.contract.schema
    split_policy = $doctor.contract.split_policy
    chunk_policy = $doctor.contract.chunk_policy
    summary = $doctor.summary
    findings = $doctor.findings
    warnings = $doctor.warnings
}

[pscustomobject]@{
    dataset = [System.IO.Path]::GetFullPath($Dataset)
    target = $Target
    data_doctor = $doctorSummary
    planner_summary = [pscustomobject]@{
        records_estimate = $plan.records_estimate
        features_estimate = $plan.features_estimate
        task = $plan.task
        data_mode = $plan.data_mode
        warnings = $plan.warnings
        sources = $plan.sources
    }
} | ConvertTo-Json -Depth 100

Write-Host "`nUX check: are target choice, warnings and the difference between bounded Doctor inspection and the planner estimate clear?"
