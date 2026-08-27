param(
    [string]$BuildDirectory = "native/build"
)

$ErrorActionPreference = "Stop"
cmake -S native -B $BuildDirectory -G "MinGW Makefiles" -DBUILD_TESTING=ON
cmake --build $BuildDirectory --config Release --parallel 4
ctest --test-dir $BuildDirectory --output-on-failure
