[CmdletBinding()]
param(
    [Parameter(Mandatory)] [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })] [string]$Dataset,
    [Parameter(Mandatory)] [string]$Target,
    [ValidateRange(10, 100)] [int]$Cpu = 80,
    [ValidateRange(10, 95)] [int]$Ram = 70,
    [ValidateRange(1, 256)] [int]$Threads = 4,
    [string]$Workspace = (Join-Path (Get-Location) ".kernelyra-user-journeys"),
    [string]$Python
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$pythonPath = if ($Python) { $Python } elseif ($env:KERNELYRA_PYTHON) { $env:KERNELYRA_PYTHON } else { (Get-Command python -CommandType Application -ErrorAction Stop).Source }
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw "Kernelyra Python was not found: $pythonPath" }
$env:KERNELYRA_PYTHON = $pythonPath
Import-Module (Join-Path $repoRoot "powershell\Kernelyra.psd1") -Force

$null = New-Item -ItemType Directory -Force -Path $Workspace
$plan = Get-KernelyraPlan -Dataset $Dataset -Target $Target -Execution cpu -Backend auto -Cpu $Cpu -Ram $Ram -Gpu 0 -Threads $Threads -Workspace $Workspace
$tuning = Get-KernelyraCpuTuning -Execution cpu -Records ([Math]::Max(32, [int]$plan.records_estimate)) -Features ([Math]::Max(1, [int]$plan.features_estimate)) -BatchSize ([Math]::Max(1, [int]$plan.batch_size)) -Workspace $Workspace
$planSummary = [pscustomobject]@{
    dataset = $plan.dataset
    target = $plan.target
    task = $plan.task
    backend = $plan.backend
    architecture = $plan.architecture
    model_format = $plan.model_format
    execution = $plan.execution
    resources = [pscustomobject]@{ cpu = $plan.cpu; ram = $plan.ram; gpu = $plan.gpu; threads = $plan.threads }
    batch_size = $plan.batch_size
    data_mode = $plan.data_mode
    records_estimate = $plan.records_estimate
    features_estimate = $plan.features_estimate
    sources = $plan.sources
    split_policy = $plan.split_policy
    chunk_policy = $plan.chunk_policy
    warnings = $plan.warnings
}

[pscustomobject]@{
    requested = [pscustomobject]@{ execution = "cpu"; cpu_percent = $Cpu; ram_percent = $Ram; threads = $Threads }
    resolved_plan = $planSummary
    tuning_advice = $tuning
} | ConvertTo-Json -Depth 100

Write-Host "`nUX check: can you see which values were requested and which became automatic?"
