[CmdletBinding()]
param(
    [Parameter(Mandatory)] [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })] [string]$Model,
    [Parameter(Mandatory)] [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })] [string]$Dataset,
    [Parameter(Mandatory)] [string]$Target,
    [ValidateSet("auto", "native", "numpy", "torch", "tensorflow")] [string]$Backend = "auto",
    [ValidateSet("auto", "cpu", "hybrid")] [string]$Execution = "cpu",
    [ValidateSet("careful", "balanced", "throughput", "maximum")] [string]$Pack = "balanced",
    [ValidateRange(1, 1000000)] [int]$MaxSteps = 100,
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
$extension = [System.IO.Path]::GetExtension($Model).ToLowerInvariant()
$capabilities = Invoke-Kernelyra -Workspace $Workspace -Arguments @("capabilities")
$preview = [pscustomobject]@{
    model = [System.IO.Path]::GetFullPath($Model)
    model_extension = $extension
    dataset = [System.IO.Path]::GetFullPath($Dataset)
    target = $Target
    requested = [pscustomobject]@{ backend = $Backend; execution = $Execution; pack = $Pack; max_steps = $MaxSteps }
    model_format_capabilities = $capabilities.model_formats
    warning = "0.5.0a3 supports specific fine-tune imports, not arbitrary model containers. Check capability roles before applying."
}

if (-not $Apply) {
    $preview | ConvertTo-Json -Depth 100
    Write-Host "`nNo fine-tune run was created. Re-run with -Apply after reviewing the input format."
    return
}

$result = Invoke-Kernelyra -Workspace $Workspace -Arguments @("finetune", $Model, $Dataset, "--target", $Target, "--backend", $Backend, "--execution", $Execution, "--pack", $Pack, "--max-steps", $MaxSteps)
[pscustomobject]@{ preview = $preview; result = $result } | ConvertTo-Json -Depth 100
Write-Host "`nUX check: was it clear which model formats are importable for fine-tuning and which are merely recognized?"
