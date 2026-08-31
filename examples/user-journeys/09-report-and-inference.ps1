[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$RunId,
    [ValidateRange(1, 10000)] [int]$Requests = 20,
    [string]$Output,
    [switch]$AutoStart,
    [string]$Workspace = (Join-Path (Get-Location) ".kernelyra-user-journeys"),
    [string]$Python
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$pythonPath = if ($Python) { $Python } elseif ($env:KERNELYRA_PYTHON) { $env:KERNELYRA_PYTHON } else { (Get-Command python -CommandType Application -ErrorAction Stop).Source }
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw "Kernelyra Python was not found: $pythonPath" }
if ($AutoStart) {
    & $pythonPath -c "import fastapi" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "-AutoStart needs the optional gateway dependency. Install it in this Python environment: $pythonPath -m pip install -e '.[gateway]'"
    }
}

function Invoke-KernelyraDaemonJson {
    param([Parameter(Mandatory)] [string[]]$Command)

    $cliArguments = @("--workspace", [System.IO.Path]::GetFullPath($Workspace), "--json")
    if ($AutoStart) { $cliArguments += "--autostart" }
    $cliArguments += $Command
    $raw = & $pythonPath -m kernelyra @cliArguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Kernelyra command failed: $($raw | Out-String)" }
    return ($raw | Out-String | ConvertFrom-Json -Depth 100)
}

$destination = if ($Output) { $Output } else { Join-Path $Workspace ("ux-report-{0}.json" -f $RunId) }
$run = Invoke-KernelyraDaemonJson -Command @("run", "get", $RunId)
if ($run.status -ne "completed") {
    throw "Run '$RunId' is '$($run.status)'. Inference verification requires a completed run."
}

$report = Invoke-KernelyraDaemonJson -Command @("report", $RunId, "--output", $destination)
$inference = Invoke-KernelyraDaemonJson -Command @("infer", $RunId, "--requests", $Requests)
$runSummary = [pscustomobject]@{
    id = $run.id
    status = $run.status
    backend = $run.effective_backend
    architecture = $run.architecture
    task = $run.objective
    best_score = $run.best_score
    checkpoint = $run.checkpoint
    termination_reason = $run.termination_reason
}
$inferenceSummary = [pscustomobject]@{
    contract = $inference.contract
    backend = $inference.backend
    task = $inference.task
    checkpoint = $inference.checkpoint
    checkpoint_immutable = $inference.checkpoint_immutable
    summary = $inference.summary
}

[pscustomobject]@{ run = $runSummary; report = $report; inference = $inferenceSummary } | ConvertTo-Json -Depth 100
if (-not $AutoStart) {
    Write-Host "`nDaemon was not started automatically. If this fails with DaemonUnavailableError, retry with -AutoStart or run 'kernelyra daemon start'."
}
Write-Host "`nUX check: did the report show how to reproduce the run, and did inference clearly state that the checkpoint remained immutable?"
