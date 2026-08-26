Set-StrictMode -Version Latest

function Get-KernelyraInvocation {
    $command = Get-Command kernelyra -ErrorAction SilentlyContinue
    if ($null -ne $command) { return @($command.Source) }
    if ($env:KERNELYRA_PYTHON) { return @($env:KERNELYRA_PYTHON, "-m", "kernelyra") }

    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $launcher) {
        foreach ($version in @("3.13", "3.12", "3.11")) {
            & $launcher.Source "-$version" -c "import kernelyra" 2>$null
            if ($LASTEXITCODE -eq 0) { return @($launcher.Source, "-$version", "-m", "kernelyra") }
        }
    }
    return @("python", "-m", "kernelyra")
}

function Invoke-KernelyraJson {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Workspace,
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    $resolvedWorkspace = [System.IO.Path]::GetFullPath($Workspace)
    $invocation = @(Get-KernelyraInvocation)
    $runner = $invocation[0]
    $runnerArgs = if ($invocation.Count -gt 1) { $invocation[1..($invocation.Count - 1)] } else { @() }
    $result = & $runner @runnerArgs --workspace $resolvedWorkspace --json @Arguments 2>&1
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
        [Parameter(Mandatory, Position = 0, ValueFromPipelineByPropertyName)] [Alias("FullName")] [string]$Path,
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
        [Parameter(Mandatory, Position = 0, ValueFromPipelineByPropertyName)] [Alias("FullName")] [string]$Dataset,
        [string]$Target,
        [ValidateSet("auto", "cpu", "hybrid")] [string]$Execution = "auto",
        [ValidateSet("careful", "balanced", "throughput", "maximum")] [string]$Pack = "balanced",
        [ValidateSet("auto", "native", "numpy", "torch", "tensorflow")]
        [string]$Backend = "auto",
        [ValidateRange(10,100)] [int]$Cpu = 70,
        [ValidateRange(10,95)] [int]$Ram = 70,
        [ValidateRange(0,100)] [int]$Gpu = 0,
        [ValidateRange(1,256)] [int]$Threads = 1,
        [string]$Workspace = "."
    )

    $arguments = @("plan", $Dataset, "--execution", $Execution, "--pack", $Pack, "--backend", $Backend, "--cpu", $Cpu, "--ram", $Ram, "--gpu", $Gpu, "--threads", $Threads)
    if ($Target) { $arguments += @("--target", $Target) }
    Invoke-KernelyraJson -Workspace $Workspace -Arguments $arguments
}

function Start-KernelyraTraining {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory, Position = 0, ValueFromPipelineByPropertyName)] [Alias("FullName")] [string]$Dataset,
        [string]$Target,
        [ValidateSet("auto", "cpu", "hybrid")] [string]$Execution = "auto",
        [ValidateSet("careful", "balanced", "throughput", "maximum")] [string]$Pack = "balanced",
        [ValidateSet("auto", "native", "numpy", "torch", "tensorflow")]
        [string]$Backend = "auto",
        [ValidateRange(10,100)] [int]$Cpu = 70,
        [ValidateRange(10,95)] [int]$Ram = 70,
        [ValidateRange(0,100)] [int]$Gpu = 0,
        [ValidateRange(1,256)] [int]$Threads = 1,
        [int]$MaxSteps = 1400,
        [int]$Seed = 42,
        [string]$Workspace = "."
    )

    $arguments = @("train", $Dataset, "--execution", $Execution, "--pack", $Pack, "--backend", $Backend, "--cpu", $Cpu, "--ram", $Ram, "--gpu", $Gpu, "--threads", $Threads, "--max-steps", $MaxSteps, "--seed", $Seed)
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
    $invocation = @(Get-KernelyraInvocation)
    $runner = $invocation[0]
    $runnerArgs = if ($invocation.Count -gt 1) { $invocation[1..($invocation.Count - 1)] } else { @() }
    & $runner @runnerArgs --workspace $resolvedWorkspace run watch $RunId
}

function Get-KernelyraExecution {
    [CmdletBinding()]
    param([string]$Workspace = ".")
    Invoke-KernelyraJson -Workspace $Workspace -Arguments @("execution")
}

function Get-KernelyraChunkPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)] [int]$Records,
        [int]$TargetRecords = 4096,
        [string]$Workspace = "."
    )
    Invoke-KernelyraJson -Workspace $Workspace -Arguments @("chunk-plan", $Records, "--target-records", $TargetRecords)
}

function Invoke-Kernelyra {
    [CmdletBinding()]
    param(
        [string]$Workspace = ".",
        [Parameter(Mandatory, ValueFromRemainingArguments)] [string[]]$Arguments
    )
    Invoke-KernelyraJson -Workspace $Workspace -Arguments $Arguments
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
    "Get-KernelyraReport",
    "Get-KernelyraExecution",
    "Get-KernelyraChunkPlan",
    "Invoke-Kernelyra"
)
