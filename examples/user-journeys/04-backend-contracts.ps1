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
$pythonPath = if ($Python) { $Python } elseif ($env:KERNELYRA_PYTHON) { $env:KERNELYRA_PYTHON } else { (Get-Command python -CommandType Application -ErrorAction Stop).Source }
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw "Kernelyra Python was not found: $pythonPath" }
$env:KERNELYRA_PYTHON = $pythonPath
Import-Module (Join-Path $repoRoot "powershell\Kernelyra.psd1") -Force

$null = New-Item -ItemType Directory -Force -Path $Workspace
$capabilities = Invoke-Kernelyra -Workspace $Workspace -Arguments @("capabilities")
$capabilitySummary = [pscustomobject]@{
    backends = @($capabilities.backends | Select-Object name, available, version, task_types, diagnostic)
    architectures = @($capabilities.architectures | Select-Object id, implemented, modalities, tasks, backends, note)
    model_formats = @($capabilities.model_formats | Select-Object id, training_output, fine_tune, architectures)
}
$contracts = @(
    [pscustomobject]@{ backend = "native"; architecture = "linear" },
    [pscustomobject]@{ backend = "numpy"; architecture = "linear" },
    [pscustomobject]@{ backend = "torch"; architecture = "mlp" },
    [pscustomobject]@{ backend = "tensorflow"; architecture = "mlp" },
    [pscustomobject]@{ backend = "native"; architecture = "mlp" },
    [pscustomobject]@{ backend = "torch"; architecture = "transformer" }
)

$attempts = foreach ($contract in $contracts) {
    try {
        $result = Invoke-Kernelyra -Workspace $Workspace -Arguments @("plan", $Dataset, "--target", $Target, "--backend", $contract.backend, "--architecture", $contract.architecture, "--execution", "cpu")
        $summary = [pscustomobject]@{
            task = $result.task
            effective_backend = $result.backend
            effective_architecture = $result.architecture
            model_format = $result.model_format
            warnings = $result.warnings
        }
        [pscustomobject]@{ backend = $contract.backend; architecture = $contract.architecture; outcome = "accepted"; detail = $summary }
    } catch {
        [pscustomobject]@{ backend = $contract.backend; architecture = $contract.architecture; outcome = "rejected"; detail = $_.Exception.Message }
    }
}

[pscustomobject]@{ capabilities = $capabilitySummary; plan_attempts = $attempts } | ConvertTo-Json -Depth 100

Write-Host "`nUX check: do accepted and rejected combinations make sense without reading source code? Rejection is expected for recognized-but-unimplemented contracts."
