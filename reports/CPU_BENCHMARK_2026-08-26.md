# CPU framework matrix — 26 August 2026

This is a reproducible CPU-only measurement of Kernelyra plus ten common ML
libraries. It is evidence for this exact workload, not a claim of universal
framework superiority.

## Environment and method

| Item | Value |
| --- | --- |
| Host | Windows 11, Intel Core i5-1235U (10 physical / 12 logical cores) |
| Python | 3.12.9 |
| GPU | Disabled; no CUDA GPU is present on the host |
| CPU cap | One thread for every runner (`KERNELYRA_BENCH_THREADS=1`) |
| Data | 8,192 train rows + 2,048 independent hold-out rows; 64 float32 features |
| Training | 30 full-data steps, learning rate 0.03, one unrecorded warm-up, three measured repetitions |
| Reported time | Median wall time for model construction, training, and hold-out prediction |

The input is a deterministic synthetic binary-classification task. The hold-out
data is drawn separately from the same hidden rule. Versions and unrounded
values are in [the raw JSON record](cpu-framework-matrix-2026-08-26.json).

## Same-task linear implementations

These six entries perform full-batch float32 logistic-regression updates on the
same data. They are the only rows that may be compared for speed.

| Framework | Version | Median time | Steps/s | Hold-out accuracy |
| --- | ---: | ---: | ---: | ---: |
| NumPy | 2.5.2 | 3.474 ms | 8,636.3 | 96.09% |
| Kernelyra native | 0.5.0a1 | 14.673 ms | 2,044.5 | 96.09% |
| PyTorch | 2.13.0 | 26.769 ms | 1,120.7 | 96.09% |
| TensorFlow | 2.21.0 | 114.419 ms | 262.2 | 96.09% |
| JAX | 0.11.1 | 207.943 ms | 144.3 | 96.09% |
| Flax + Optax | 0.12.9 / 0.2.8 | 290.078 ms | 103.4 | 96.09% |

For this small, single-threaded linear workload, NumPy is 4.22× faster than
Kernelyra native; Kernelyra native is 1.82× faster than PyTorch. All six reach
the same 96.09% hold-out accuracy. These ratios do **not** transfer to larger
models, GPU execution, data pipelines, mixed precision, or other tasks.

## Other library families

The rows below are useful capability measurements, but their algorithms and
update semantics differ from full-batch logistic regression. Do not use their
times to rank them against the table above.

| Framework | Version | Family | Median time | Hold-out accuracy |
| --- | ---: | --- | ---: | ---: |
| scikit-learn | 1.9.0 | SGD linear classifier | 147.890 ms | 98.24% |
| River | 0.26.1 | Online logistic regression | 9.308 s | 98.63% |
| XGBoost | 3.4.1 | 30 histogram trees | 432.822 ms | 76.37% |
| LightGBM | 4.7.0 | 30 trees | 264.585 ms | 77.25% |
| CatBoost | 1.2.10 | 30 trees | 690.095 ms | 79.49% |

The tree settings are intentionally shallow and not tuned; their lower scores
are not a quality judgement on those libraries. River performs many Python-level
online updates, so its timing measures a different operating model.

## Reproduce

```powershell
$env:KERNELYRA_BENCH_THREADS = '1'
.\.benchmarks\frameworks-cpu-2026-08-26\Scripts\python.exe `
  scripts\benchmark_tabular_frameworks.py `
  --rows 8192 --evaluation-rows 2048 --features 64 --steps 30 `
  --runs 3 --warmup-runs 1 `
  --output .benchmarks\cpu-framework-matrix.json
```

The runner saves machine-readable data and reports missing or errored packages;
it never substitutes values. A separate CUDA GPU matrix is required before
making GPU performance statements.
