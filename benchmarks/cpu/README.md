# Kernelyra dense CPU benchmark

This directory contains one maintained CPU benchmark and its raw evidence. It
is intentionally narrow: full-batch float32 binary logistic training plus
fused tabular preprocessing on one Windows laptop. It does not establish
universal framework, dataset, memory or model-quality superiority.

The runner forces common BLAS/OpenMP environment variables to one thread before
NumPy is imported. For training, both implementations receive the same matrix,
labels, zero parameters, learning rate and number of updates. Only the final
bulk step calculates loss, and execution order alternates between
implementations. For preprocessing, both implementations copy the same matrix,
impute deliberately injected non-finite values, normalize with the same
float32 statistics, and clip to the same interval.

The runner also records a separate, untimed random-batch capability probe. A
successful probe sees the C ABI, C++ dispatcher, Rust sampling policy, Fortran
numeric update and Zig memory gather in the native model trace. It is not a
speed measurement and is not added to either timing result.

Reproduce from an installed source checkout:

```powershell
cmake -S native -B native/build -G "MinGW Makefiles" -DBUILD_TESTING=ON
cmake --build native/build --config Release -j 4
$env:PYTHONPATH = (Resolve-Path .\src)
py -3.12 benchmarks\cpu\benchmark_dense_binary.py `
  --core native\build\kernelyra_core.dll `
  --output benchmark-result.json `
  --host-label "your machine" `
  --cpu-label "your CPU"
```

Published 0.6.0a1 evidence:

- [`results/modern-14-c12m-v0.6.0a1.json`](results/modern-14-c12m-v0.6.0a1.json)

The JSON contains every measured duration, exact runtime versions, thread
settings, workload parameters, final loss and accuracy, preprocessing checksum
and repair count, and the separately recorded execution trace.
