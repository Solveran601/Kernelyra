Set-StrictMode -Version Latest

function Invoke-KernelyraJson {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Workspace,
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    $resolvedWorkspace = [System.IO.Path]::GetFullPath($Workspace)
    $command = Get-Command kernelyra -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        $result = & $command.Source --workspace $resolvedWorkspace --json @Arguments 2>&1
    }
    else {
        $result = & python -m kernelyra --workspace $resolvedWorkspace --json @Arguments 2>&1
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Kernelyra command failed: $($result | Out-String)"
    }
    return ($result | Out-String | ConvertFrom-Json -Depth 100)
}

function New-KernelyraProject {
    [CmdletBinding(SupportsShouldProcess)]
    param([string]$Path = ".")

    $workspace = [System.IO.Path]::GetFullPath($Path)
    if ($PSCmdlet.ShouldProcess($workspace, "Initialize Kernelyra workspace")) {
        New-Item -ItemType Directory -Force -Path $workspace | Out-Null
        Invoke-KernelyraJson -Workspace $workspace -Arguments @("doctor")
    }
}

function Test-KernelyraDataset {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)] [string]$Path,
        [string]$Target,
        [string]$Workspace = "."
    )

    $arguments = @("dataset", "doctor", $Path)
    if ($Target) { $arguments += @("--target", $Target) }
    Invoke-KernelyraJson -Workspace $Workspace -Arguments $arguments
}

function Get-KernelyraPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)] [string]$Dataset,
        [string]$Target,
        [ValidateSet("auto", "eco", "low-memory", "balanced", "performance", "workstation", "custom")]
        [string]$Profile = "auto",
        [ValidateSet("auto", "native", "numpy", "torch", "tensorflow")]
        [string]$Backend = "auto",
        [string]$Workspace = "."
    )

    $arguments = @("plan", $Dataset, "--profile", $Profile, "--backend", $Backend)
    if ($Target) { $arguments += @("--target", $Target) }
    Invoke-KernelyraJson -Workspace $Workspace -Arguments $arguments
}

function Start-KernelyraTraining {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory, Position = 0)] [string]$Dataset,
        [string]$Target,
        [ValidateSet("auto", "eco", "low-memory", "balanced", "performance", "workstation", "custom")]
        [string]$Profile = "auto",
        [ValidateSet("auto", "native", "numpy", "torch", "tensorflow")]
        [string]$Backend = "auto",
        [int]$MaxSteps = 1400,
        [int]$Seed = 42,
        [string]$Workspace = "."
    )

    $arguments = @("train", $Dataset, "--profile", $Profile, "--backend", $Backend, "--max-steps", $MaxSteps, "--seed", $Seed)
    if ($Target) { $arguments += @("--target", $Target) }
    if ($PSCmdlet.ShouldProcess($Dataset, "Train Kernelyra model")) {
        Invoke-KernelyraJson -Workspace $Workspace -Arguments $arguments
    }
}

function Watch-KernelyraRun {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)] [string]$RunId,
        [string]$Workspace = "."
    )

    $resolvedWorkspace = [System.IO.Path]::GetFullPath($Workspace)
    & kernelyra --workspace $resolvedWorkspace run watch $RunId
}

function Resume-KernelyraRun {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory, Position = 0)] [string]$RunId,
        [string]$Workspace = "."
    )

    if ($PSCmdlet.ShouldProcess($RunId, "Resume Kernelyra run")) {
        Invoke-KernelyraJson -Workspace $Workspace -Arguments @("run", "resume", $RunId)
    }
}

function Export-KernelyraModel {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory, Position = 0)] [string]$RunId,
        [Parameter(Mandatory)] [string]$Output,
        [string]$Workspace = "."
    )

    if ($PSCmdlet.ShouldProcess($RunId, "Export Kernelyra model")) {
        Invoke-KernelyraJson -Workspace $Workspace -Arguments @("run", "export", $RunId, "--output", $Output)
    }
}

function Get-KernelyraReport {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)] [string]$RunId,
        [string]$Output,
        [string]$Workspace = "."
    )

    $arguments = @("report", $RunId)
    if ($Output) { $arguments += @("--output", $Output) }
    Invoke-KernelyraJson -Workspace $Workspace -Arguments $arguments
}

Export-ModuleMember -Function @(
    "New-KernelyraProject",
    "Test-KernelyraDataset",
    "Get-KernelyraPlan",
    "Start-KernelyraTraining",
    "Watch-KernelyraRun",
    "Resume-KernelyraRun",
    "Export-KernelyraModel",
    "Get-KernelyraReport"
)
