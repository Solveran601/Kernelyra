@{
    RootModule = 'Kernelyra.psm1'
    ModuleVersion = '0.5.3'
    GUID = '3d0c17b9-37f3-4f58-b6a8-20a43f971df3'
    Author = 'Kernelyra contributors'
    CompanyName = 'Kernelyra'
    Copyright = '(c) Kernelyra contributors'
    Description = 'PowerShell commands for local Kernelyra tabular training, native diagnostics and data contracts.'
    PowerShellVersion = '7.0'
    FunctionsToExport = @(
        'New-KernelyraProject',
        'Test-KernelyraDataset',
        'Get-KernelyraPlan',
        'Start-KernelyraTraining',
        'Watch-KernelyraRun',
        'Resume-KernelyraRun',
        'Export-KernelyraModel',
        'Get-KernelyraReport',
        'Get-KernelyraExecution',
        'Get-KernelyraNativeStatus',
        'Get-KernelyraCpuTuning',
        'Get-KernelyraDataContract',
        'Get-KernelyraPack',
        'Get-KernelyraPackAlgorithm',
        'Get-KernelyraPackTablePath',
        'Copy-KernelyraPack',
        'Add-KernelyraPackAlgorithm',
        'Remove-KernelyraPackAlgorithm',
        'Remove-KernelyraPack',
        'Get-KernelyraRunStatus',
        'Get-KernelyraChunkPlan',
        'Invoke-Kernelyra'
    )
    CmdletsToExport = @()
    VariablesToExport = @()
    AliasesToExport = @()
    PrivateData = @{
        PSData = @{
            Tags = @('Kernelyra', 'ML', 'tabular', 'training')
            ProjectUri = 'https://github.com/Solveran601/Kernelyra'
            LicenseUri = 'https://github.com/Solveran601/Kernelyra/blob/main/LICENSE'
        }
    }
}
