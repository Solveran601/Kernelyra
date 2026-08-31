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
        [string]$Workspace,
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    $invocation = @(Get-KernelyraInvocation)
    $runner = $invocation[0]
    $runnerArgs = if ($invocation.Count -gt 1) { $invocation[1..($invocation.Count - 1)] } else { @() }
    $workspaceArguments = @()
    if ($Workspace) {
        $workspaceArguments = @("--workspace", [System.IO.Path]::GetFullPath($Workspace))
    }
    $result = & $runner @runnerArgs @workspaceArguments --json @Arguments 2>&1
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
        [string]$Workspace
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
        [ValidateSet("auto", "native", "numpy", "torch", "tensorflow")]
        [string]$Backend = "auto",
        [ValidateSet("auto", "memory", "stream")] [string]$DataMode = "auto",
        [ValidateSet("none", "last", "best")] [string]$CheckpointResume = "none",
        [ValidateSet("none", "best", "last")] [string]$CheckpointFinal = "none",
        [ValidateSet("none", "best")] [string]$CheckpointRollback = "none",
        [ValidateRange(0,95)] [int]$ValidationPercent = 15,
        [ValidateRange(0,95)] [int]$TestPercent = 15,
        [string]$GroupColumn,
        [ValidateRange(128,262144)] [int]$ChunkTargetRecords,
        [ValidateRange(1,262144)] [int]$ChunkMinimumRecords,
        [ValidateRange(1,262144)] [int]$ChunkMaximumRecords,
        [ValidateRange(10,100)] [int]$Cpu = 70,
        [ValidateRange(10,95)] [int]$Ram = 70,
        [ValidateRange(0,100)] [int]$Gpu = 0,
        [ValidateRange(1,256)] [Nullable[int]]$Threads,
        [ValidateRange(0,64)] [Nullable[int]]$DataWorkers,
        [ValidateRange(0,32)] [Nullable[int]]$Prefetch,
        [string]$Workspace
    )

    $arguments = @("plan", $Dataset, "--execution", $Execution, "--backend", $Backend, "--data-mode", $DataMode, "--checkpoint-resume", $CheckpointResume, "--checkpoint-final", $CheckpointFinal, "--checkpoint-rollback", $CheckpointRollback, "--validation-percent", $ValidationPercent, "--test-percent", $TestPercent, "--cpu", $Cpu, "--ram", $Ram, "--gpu", $Gpu)
    if ($Target) { $arguments += @("--target", $Target) }
    if ($GroupColumn) { $arguments += @("--group-column", $GroupColumn) }
    if ($PSBoundParameters.ContainsKey("ChunkTargetRecords")) { $arguments += @("--chunk-target-records", $ChunkTargetRecords) }
    if ($PSBoundParameters.ContainsKey("ChunkMinimumRecords")) { $arguments += @("--chunk-minimum-records", $ChunkMinimumRecords) }
    if ($PSBoundParameters.ContainsKey("ChunkMaximumRecords")) { $arguments += @("--chunk-maximum-records", $ChunkMaximumRecords) }
    if ($PSBoundParameters.ContainsKey("Threads")) { $arguments += @("--threads", $Threads) }
    if ($PSBoundParameters.ContainsKey("DataWorkers")) { $arguments += @("--data-workers", $DataWorkers) }
    if ($PSBoundParameters.ContainsKey("Prefetch")) { $arguments += @("--prefetch", $Prefetch) }
    Invoke-KernelyraJson -Workspace $Workspace -Arguments $arguments
}

function Start-KernelyraTraining {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory, Position = 0, ValueFromPipelineByPropertyName)] [Alias("FullName")] [string]$Dataset,
        [string]$Target,
        [ValidateSet("auto", "cpu", "hybrid")] [string]$Execution = "auto",
        [ValidateSet("auto", "native", "numpy", "torch", "tensorflow")]
        [string]$Backend = "auto",
        [ValidateSet("auto", "memory", "stream")] [string]$DataMode = "auto",
        [ValidateSet("none", "last", "best")] [string]$CheckpointResume = "none",
        [ValidateSet("none", "best", "last")] [string]$CheckpointFinal = "none",
        [ValidateSet("none", "best")] [string]$CheckpointRollback = "none",
        [ValidateRange(0,95)] [int]$ValidationPercent = 15,
        [ValidateRange(0,95)] [int]$TestPercent = 15,
        [string]$GroupColumn,
        [ValidateRange(128,262144)] [int]$ChunkTargetRecords,
        [ValidateRange(1,262144)] [int]$ChunkMinimumRecords,
        [ValidateRange(1,262144)] [int]$ChunkMaximumRecords,
        [ValidateRange(10,100)] [int]$Cpu = 70,
        [ValidateRange(10,95)] [int]$Ram = 70,
        [ValidateRange(0,100)] [int]$Gpu = 0,
        [ValidateRange(1,256)] [Nullable[int]]$Threads,
        [ValidateRange(0,64)] [Nullable[int]]$DataWorkers,
        [ValidateRange(0,32)] [Nullable[int]]$Prefetch,
        [ValidateRange(1,10000000)] [int]$MaxSteps = 1400,
        [int]$Seed = 42,
        [string]$Workspace
    )

    $arguments = @("train", $Dataset, "--execution", $Execution, "--backend", $Backend, "--data-mode", $DataMode, "--checkpoint-resume", $CheckpointResume, "--checkpoint-final", $CheckpointFinal, "--checkpoint-rollback", $CheckpointRollback, "--validation-percent", $ValidationPercent, "--test-percent", $TestPercent, "--cpu", $Cpu, "--ram", $Ram, "--gpu", $Gpu, "--max-steps", $MaxSteps, "--seed", $Seed)
    if ($Target) { $arguments += @("--target", $Target) }
    if ($GroupColumn) { $arguments += @("--group-column", $GroupColumn) }
    if ($PSBoundParameters.ContainsKey("ChunkTargetRecords")) { $arguments += @("--chunk-target-records", $ChunkTargetRecords) }
    if ($PSBoundParameters.ContainsKey("ChunkMinimumRecords")) { $arguments += @("--chunk-minimum-records", $ChunkMinimumRecords) }
    if ($PSBoundParameters.ContainsKey("ChunkMaximumRecords")) { $arguments += @("--chunk-maximum-records", $ChunkMaximumRecords) }
    if ($PSBoundParameters.ContainsKey("Threads")) { $arguments += @("--threads", $Threads) }
    if ($PSBoundParameters.ContainsKey("DataWorkers")) { $arguments += @("--data-workers", $DataWorkers) }
    if ($PSBoundParameters.ContainsKey("Prefetch")) { $arguments += @("--prefetch", $Prefetch) }
    if ($PSCmdlet.ShouldProcess($Dataset, "Train Kernelyra model")) {
        Invoke-KernelyraJson -Workspace $Workspace -Arguments $arguments
    }
}

function Watch-KernelyraRun {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)] [string]$RunId,
        [string]$Workspace
    )

    $invocation = @(Get-KernelyraInvocation)
    $runner = $invocation[0]
    $runnerArgs = if ($invocation.Count -gt 1) { $invocation[1..($invocation.Count - 1)] } else { @() }
    $workspaceArguments = @()
    if ($Workspace) {
        $workspaceArguments = @("--workspace", [System.IO.Path]::GetFullPath($Workspace))
    }
    & $runner @runnerArgs @workspaceArguments run watch $RunId
}

function Get-KernelyraExecution {
    [CmdletBinding()]
    param([string]$Workspace)
    Invoke-KernelyraJson -Workspace $Workspace -Arguments @("execution")
}

function Get-KernelyraNativeStatus {
    [CmdletBinding()]
    param([string]$Workspace)

    Invoke-KernelyraJson -Workspace $Workspace -Arguments @("native", "status")
}

function Get-KernelyraCpuTuning {
    [CmdletBinding()]
    param(
        [ValidateSet("auto", "cpu", "hybrid")] [string]$Execution = "cpu",
        [ValidateRange(32,1000000000)] [int]$Records = 100000,
        [ValidateRange(1,1000000)] [int]$Features = 32,
        [ValidateRange(1,1000000)] [int]$BatchSize = 64,
        [switch]$Streaming,
        [string]$Workspace
    )

    $arguments = @("tune", "--execution", $Execution, "--records", $Records, "--features", $Features, "--batch-size", $BatchSize)
    if ($Streaming) { $arguments += "--streaming" }
    Invoke-KernelyraJson -Workspace $Workspace -Arguments $arguments
}

function Get-KernelyraDataContract {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0, ValueFromPipelineByPropertyName)] [Alias("FullName")] [string]$Path,
        [string]$Target,
        [string]$Workspace
    )

    $arguments = @("dataset", "doctor", $Path)
    if ($Target) { $arguments += @("--target", $Target) }
    Invoke-KernelyraJson -Workspace $Workspace -Arguments $arguments
}

function Get-KernelyraRunStatus {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)] [string]$RunId,
        [string]$Workspace
    )

    Invoke-KernelyraJson -Workspace $Workspace -Arguments @("run", "get", $RunId)
}

function Get-KernelyraChunkPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)] [int]$Records,
        [int]$TargetRecords = 4096,
        [ValidateRange(0,95)] [int]$ValidationPercent = 15,
        [ValidateRange(0,95)] [int]$TestPercent = 15,
        [string]$Workspace
    )
    Invoke-KernelyraJson -Workspace $Workspace -Arguments @("chunk-plan", $Records, "--target-records", $TargetRecords, "--validation-percent", $ValidationPercent, "--test-percent", $TestPercent)
}

function Invoke-Kernelyra {
    [CmdletBinding()]
    param(
        [string]$Workspace,
        [Parameter(Mandatory, ValueFromRemainingArguments)] [string[]]$Arguments
    )
    Invoke-KernelyraJson -Workspace $Workspace -Arguments $Arguments
}

function Resume-KernelyraRun {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory, Position = 0)] [string]$RunId,
        [string]$Workspace
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
        [string]$Workspace
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
        [string]$Workspace
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
    "Get-KernelyraNativeStatus",
    "Get-KernelyraCpuTuning",
    "Get-KernelyraDataContract",
    "Get-KernelyraRunStatus",
    "Get-KernelyraChunkPlan",
    "Invoke-Kernelyra"
)
