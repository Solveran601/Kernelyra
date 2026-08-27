# Kernelyra dense CPU benchmark

This directory contains one maintained microbenchmark and its raw evidence.
It is intentionally narrow: full-batch float32 binary logistic training on one
Windows laptop. It does not establish universal framework, dataset, memory or
model-quality superiority.

The runner forces common BLAS/OpenMP environment variables to one thread before
NumPy is imported. Both implementations receive the same matrix, labels, zero
parameters, learning rate and number of updates. Only the final bulk step
calculates loss, and execution order alternates between implementations.

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

Published evidence:

- [`results/modern-14-c12m-v0.5.0a3.json`](results/modern-14-c12m-v0.5.0a3.json)

The JSON contains every measured duration, exact runtime versions, thread
settings, workload parameters, final loss and accuracy.
