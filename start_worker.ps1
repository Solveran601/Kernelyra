$ErrorActionPreference = 'Stop'
$candidates = @()
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $venvPython) {
  $candidates += [pscustomobject]@{ Path = $venvPython; Prefix = @() }
}
foreach ($name in @('py', 'python', 'python3')) {
  $command = Get-Command $name -ErrorAction SilentlyContinue
  if ($command) {
    $prefix = if ($name -eq 'py') { @('-3') } else { @() }
    $candidates += [pscustomobject]@{ Path = $command.Source; Prefix = $prefix }
  }
}

$selected = $null
foreach ($candidate in $candidates) {
  & $candidate.Path @($candidate.Prefix) -c 'import numpy' 2>$null
  if ($LASTEXITCODE -eq 0) { $selected = $candidate; break }
}
if (-not $selected) {
  throw "Python 3.11-3.13 with NumPy was not found. Run: python -m pip install -e `"$PSScriptRoot`""
}

$worker = Join-Path $PSScriptRoot 'worker.py'
& $selected.Path @($selected.Prefix) -B $worker @args
exit $LASTEXITCODE
