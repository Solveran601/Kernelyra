[CmdletBinding()]
param(
    [Parameter(Mandatory)] [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })] [string]$Dataset,
    [Parameter(Mandatory)] [string]$Target,
    [ValidateRange(1, 1000000)] [int]$MaxSteps = 100,
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
$result = Start-KernelyraTraining -Dataset $Dataset -Target $Target -Execution cpu -Backend native -Cpu $Cpu -Ram $Ram -Gpu 0 -Threads $Threads -MaxSteps $MaxSteps -Workspace $Workspace
$summary = [pscustomobject]@{
    checkpoint = $result.checkpoint
    dataset = [pscustomobject]@{ id = $result.dataset.id; records = $result.dataset.records; features = $result.dataset.features; warnings = $result.dataset.warnings }
    run = [pscustomobject]@{
        id = $result.run.id
        status = $result.run.status
        backend = $result.run.effective_backend
        architecture = $result.run.architecture
        task = $result.run.objective
        best_score = $result.run.best_score
        best_step = $result.run.best_step
        termination_reason = $result.run.termination_reason
        metrics = [pscustomobject]@{
            train = $result.run.metrics.train
            validation = $result.run.metrics.validation
            test = $result.run.metrics.test
            health = $result.run.metrics.health
        }
    }
    plan = [pscustomobject]@{ batch_size = $result.plan.batch_size; sources = $result.plan.sources; warnings = $result.plan.warnings }
}
$summary | ConvertTo-Json -Depth 100

Write-Host "`nUX check: copy the run id and checkpoint path from this result. Then use 09-report-and-inference.ps1 on the completed run."
