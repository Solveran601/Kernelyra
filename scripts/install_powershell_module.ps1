[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$Destination = (Join-Path $env:USERPROFILE 'Documents\PowerShell\Modules\Kernelyra\0.5.1')
)

$source = Join-Path $PSScriptRoot '..\powershell'
$resolvedSource = [System.IO.Path]::GetFullPath($source)
$resolvedDestination = [System.IO.Path]::GetFullPath($Destination)
if ($PSCmdlet.ShouldProcess($resolvedDestination, 'Install local Kernelyra PowerShell module')) {
    New-Item -ItemType Directory -Force -Path $resolvedDestination | Out-Null
    Copy-Item -Path (Join-Path $resolvedSource '*') -Destination $resolvedDestination -Force
    Write-Output "Installed Kernelyra PowerShell module to $resolvedDestination"
}
