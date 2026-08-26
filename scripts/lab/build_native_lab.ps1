param(
    [string]$BuildDirectory = "native/build",
    [switch]$Test
)

$ErrorActionPreference = "Stop"
cmake -S native -B $BuildDirectory -G "MinGW Makefiles" -DBUILD_TESTING=ON
cmake --build $BuildDirectory --config Release --parallel 4
if ($Test) {
    ctest --test-dir $BuildDirectory --output-on-failure
}
Write-Output "Lab DLL: $(Resolve-Path (Join-Path $BuildDirectory 'kernelyra_core.dll'))"
