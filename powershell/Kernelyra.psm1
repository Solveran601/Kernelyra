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
        [ValidatePattern("^[a-z0-9][a-z0-9._-]{0,63}$")] [string]$Pack = "balanced",
        [ValidateSet("auto", "native", "numpy", "torch", "tensorflow")]
        [string]$Backend = "auto",
        [ValidateSet("auto", "memory", "stream")] [string]$DataMode = "auto",
        [ValidateRange(10,100)] [int]$Cpu = 70,
        [ValidateRange(10,95)] [int]$Ram = 70,
        [ValidateRange(0,100)] [int]$Gpu = 0,
        [ValidateRange(1,256)] [int]$Threads = 1,
        [string]$Workspace = "."
    )

    $arguments = @("plan", $Dataset, "--execution", $Execution, "--pack", $Pack, "--backend", $Backend, "--data-mode", $DataMode, "--cpu", $Cpu, "--ram", $Ram, "--gpu", $Gpu, "--threads", $Threads)
    if ($Target) { $arguments += @("--target", $Target) }
    Invoke-KernelyraJson -Workspace $Workspace -Arguments $arguments
}

function Start-KernelyraTraining {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory, Position = 0, ValueFromPipelineByPropertyName)] [Alias("FullName")] [string]$Dataset,
        [string]$Target,
        [ValidateSet("auto", "cpu", "hybrid")] [string]$Execution = "auto",
        [ValidatePattern("^[a-z0-9][a-z0-9._-]{0,63}$")] [string]$Pack = "balanced",
        [ValidateSet("auto", "native", "numpy", "torch", "tensorflow")]
        [string]$Backend = "auto",
        [ValidateSet("auto", "memory", "stream")] [string]$DataMode = "auto",
        [ValidateRange(10,100)] [int]$Cpu = 70,
        [ValidateRange(10,95)] [int]$Ram = 70,
        [ValidateRange(0,100)] [int]$Gpu = 0,
        [ValidateRange(1,256)] [int]$Threads = 1,
        [int]$MaxSteps = 1400,
        [int]$Seed = 42,
        [string]$Workspace = "."
    )

    $arguments = @("train", $Dataset, "--execution", $Execution, "--pack", $Pack, "--backend", $Backend, "--data-mode", $DataMode, "--cpu", $Cpu, "--ram", $Ram, "--gpu", $Gpu, "--threads", $Threads, "--max-steps", $MaxSteps, "--seed", $Seed)
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

function Get-KernelyraNativeStatus {
    [CmdletBinding()]
    param([string]$Workspace = ".")

    Invoke-KernelyraJson -Workspace $Workspace -Arguments @("native", "status")
}

function Get-KernelyraCpuTuning {
    [CmdletBinding()]
    param(
        [ValidateSet("auto", "cpu", "hybrid")] [string]$Execution = "cpu",
        [ValidatePattern("^[a-z0-9][a-z0-9._-]{0,63}$")] [string]$Pack = "balanced",
        [ValidateRange(32,1000000000)] [int]$Records = 100000,
        [ValidateRange(1,1000000)] [int]$Features = 32,
        [ValidateRange(1,1000000)] [int]$BatchSize = 64,
        [switch]$Streaming,
        [string]$Workspace = "."
    )

    $arguments = @("tune", "--execution", $Execution, "--pack", $Pack, "--records", $Records, "--features", $Features, "--batch-size", $BatchSize)
    if ($Streaming) { $arguments += "--streaming" }
    Invoke-KernelyraJson -Workspace $Workspace -Arguments $arguments
}

function Get-KernelyraDataContract {
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

function Get-KernelyraPack {
    [CmdletBinding()]
    param(
        [Parameter(Position = 0)] [string]$Name,
        [string]$Workspace = "."
    )

    if ($Name) {
        return Invoke-KernelyraJson -Workspace $Workspace -Arguments @("packs", "show", $Name)
    }
    $result = Invoke-KernelyraJson -Workspace $Workspace -Arguments @("packs", "list")
    return $result.packs
}

function Get-KernelyraPackAlgorithm {
    [CmdletBinding()]
    param([string]$Workspace = ".")

    $result = Invoke-KernelyraJson -Workspace $Workspace -Arguments @("packs", "algorithms")
    return $result.algorithms
}

function Get-KernelyraPackTablePath {
    [CmdletBinding()]
    param([string]$Workspace = ".")

    $result = Invoke-KernelyraJson -Workspace $Workspace -Arguments @("packs", "path")
    return $result.path
}

function Copy-KernelyraPack {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory, Position = 0)]
        [ValidatePattern("^[a-z0-9][a-z0-9._-]{0,63}$")] [string]$Name,
        [Alias("From")]
        [ValidatePattern("^[a-z0-9][a-z0-9._-]{0,63}$")] [string]$Base = "balanced",
        [string]$Label,
        [string]$Workspace = "."
    )

    $arguments = @("packs", "clone", $Name, "--from", $Base)
    if ($Label) { $arguments += @("--label", $Label) }
    if ($PSCmdlet.ShouldProcess($Name, "Create custom Kernelyra pack from $Base")) {
        Invoke-KernelyraJson -Workspace $Workspace -Arguments $arguments
    }
}

function Add-KernelyraPackAlgorithm {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory, Position = 0)] [string]$Name,
        [Parameter(Mandatory, Position = 1)] [string]$Algorithm,
        [string]$Workspace = "."
    )

    if ($PSCmdlet.ShouldProcess($Name, "Add Kernelyra pack algorithm $Algorithm")) {
        Invoke-KernelyraJson -Workspace $Workspace -Arguments @("packs", "add-algorithm", $Name, $Algorithm)
    }
}

function Remove-KernelyraPackAlgorithm {
    [CmdletBinding(SupportsShouldProcess, ConfirmImpact = "Medium")]
    param(
        [Parameter(Mandatory, Position = 0)] [string]$Name,
        [Parameter(Mandatory, Position = 1)] [string]$Algorithm,
        [string]$Workspace = "."
    )

    if ($PSCmdlet.ShouldProcess($Name, "Remove Kernelyra pack algorithm $Algorithm")) {
        Invoke-KernelyraJson -Workspace $Workspace -Arguments @("packs", "remove-algorithm", $Name, $Algorithm)
    }
}

function Remove-KernelyraPack {
    [CmdletBinding(SupportsShouldProcess, ConfirmImpact = "High")]
    param(
        [Parameter(Mandatory, Position = 0)] [string]$Name,
        [string]$Workspace = "."
    )

    if ($PSCmdlet.ShouldProcess($Name, "Delete custom Kernelyra pack")) {
        Invoke-KernelyraJson -Workspace $Workspace -Arguments @("packs", "delete", $Name)
    }
}

function Get-KernelyraRunStatus {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)] [string]$RunId,
        [string]$Workspace = "."
    )

    Invoke-KernelyraJson -Workspace $Workspace -Arguments @("run", "get", $RunId)
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
    "Get-KernelyraNativeStatus",
    "Get-KernelyraCpuTuning",
    "Get-KernelyraDataContract",
    "Get-KernelyraPack",
    "Get-KernelyraPackAlgorithm",
    "Get-KernelyraPackTablePath",
    "Copy-KernelyraPack",
    "Add-KernelyraPackAlgorithm",
    "Remove-KernelyraPackAlgorithm",
    "Remove-KernelyraPack",
    "Get-KernelyraRunStatus",
    "Get-KernelyraChunkPlan",
    "Invoke-Kernelyra"
)
