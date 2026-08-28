<p align="center">
  <img src="assets/brand/kernelyra-logo.png" alt="Kernelyra" width="520">
</p>

<p align="center">
  <a href="https://github.com/Solveran601/Kernelyra/actions"><img src="https://img.shields.io/github/actions/workflow/status/Solveran601/Kernelyra/ci.yml?branch=main&label=CI" alt="CI status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0 license"></a>
  <img src="https://img.shields.io/badge/status-alpha-f59e0b" alt="Alpha status">
  <a href="README.ru.md">Русская версия</a>
</p>

<p align="center"><img src="assets/brand/kernelyra-mark-animated.svg" alt="Animated Kernelyra mark" width="72"></p>

<p align="center"><strong>Native-first, resource-aware training for tabular data.</strong></p>

Kernelyra **0.6.0a1 (alpha)** is a local terminal-first library for
tabular model training. The same planning path is available through the CLI,
Python API, PowerShell module, and JSONL protocol used by the bundled SDKs.

<p align="center">
  <a href="#capabilities">Capabilities</a> ·
  <a href="#install">Install</a> ·
  <a href="#powershell">PowerShell</a> ·
  <a href="#limits">Limits</a>
</p>

<a id="capabilities"></a>

## What works in this alpha

- Binary and multiclass classification, plus regression, on tabular data.
- Direct training from CSV, TSV, JSONL/NDJSON, numeric NPZ, and optional
  Parquet input.
- Bundled native and NumPy backends; optional PyTorch and TensorFlow/Keras
  backends when installed. The native core has five observable roles: a C ABI
  with checked-size guards, C++ dispatch, Rust policy, Fortran numeric kernels,
  and Zig memory kernels.
  A model's `native_execution` trace reports engines that actually participated
  in its native calls; it never guesses that every engine ran.
- Explicit `cpu` or `hybrid` execution, developer-set CPU/RAM/GPU/thread
  limits, four built-in algorithm packs (`careful`, `balanced`, `throughput`,
  `maximum`), and validated user packs cloned from them. Packs alter real
  thread, bulk-dispatch, chunk, prefetch, and arena tuning; they do not
  classify the user's PC. Checkpoints resume from `last` by default; users can
  choose `best` or `last` for the final evaluation, while Model Guard restores
  only `best`.
- Data Doctor: bounded preflight findings, a signed dataset contract, a
  deterministic split recommendation, and a variable-range chunk plan.
- UTF-8 text preparation: native chunk planning when the native core is
  available, plus a reversible byte tokenizer and causal loss masks for a
  future trainer. This is not an LLM trainer.
- Model Guard V2: finite-metric checks plus saved score-trend evidence in the
  run health record.
- Portable JSON/HTML experiment reports and explicit CPU/RAM/GPU budgets.

Data Doctor is intentionally bounded: its findings describe the inspected
sample, not every row in a dataset. For materialized training, classification
uses deterministic stratification and a detected time-like column preserves
the supplied input order. For CSV, TSV, JSONL/NDJSON, and Parquet, AutoTrainer
routes a detected group/context column through the streaming splitter: each
context stays in one split and that identifier is excluded from features.
Formats without that streaming path still receive an explicit leakage warning.

<a id="install"></a>

## Install from source

PyPI publication and prebuilt GitHub releases are not published yet. Install
from a source checkout with Python 3.11–3.13.

```powershell
git clone https://github.com/Solveran601/Kernelyra.git
Set-Location Kernelyra
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -e .
```

Install `.[data]` for Parquet, `.[torch]` for PyTorch, or `.[tensorflow]` for
TensorFlow/Keras.

<a id="powershell"></a>

## CLI from PowerShell

```powershell
python -m kernelyra version
python -m kernelyra doctor
python -m kernelyra execution
python -m kernelyra plan .\data\train.csv --target label --execution cpu --pack throughput --cpu 100 --ram 85 --threads 12
```

Before training, inspect bounded data-health evidence explicitly when useful:

```powershell
python -m kernelyra dataset doctor .\data\train.csv --target label
```

List the pack table, inspect the available algorithms, then make a modified
pack without changing an immutable built-in default:

```powershell
python -m kernelyra packs list
python -m kernelyra packs algorithms
python -m kernelyra packs clone my-careful --from careful
python -m kernelyra packs add-algorithm my-careful thread_parallel_gradient
python -m kernelyra tune --execution cpu --pack my-careful --records 100000 --features 64 --batch-size 128
```

`python -m kernelyra packs path` prints the editable JSON table. Kernelyra
validates that file on every read. Allocation bounds, context boundaries, and
Model Guard are safety invariants and cannot be removed from a pack.

## Python

```python
from kernelyra import Engine

with Engine("./project") as kernelyra:
    health = kernelyra.doctor("train.csv", "label")
    result = kernelyra.fit("train.csv", "label", backend="auto")
    report = kernelyra.report(result.run.id, "./project/run-report.json")

print(report["output"])
```

For PowerShell command names, import the bundled module from a checkout:

```powershell
Import-Module .\powershell\Kernelyra.psd1 -Force
Test-KernelyraDataset .\data\train.csv -Target label
Get-KernelyraDataContract .\data\train.csv -Target label
Get-KernelyraNativeStatus
Get-KernelyraCpuTuning -Records 100000 -Features 32 -BatchSize 64
Get-KernelyraPack | Format-Table Name, Base, Built_In, Algorithms
Copy-KernelyraPack my-careful -Base careful
Add-KernelyraPackAlgorithm my-careful thread_parallel_gradient
Get-KernelyraCpuTuning -Pack my-careful -Records 100000 -Features 64 -BatchSize 128
Get-KernelyraPlan .\data\train.csv -Target label
Start-KernelyraTraining .\data\train.csv -Target label -Execution cpu -Pack throughput -Cpu 100 -Ram 85 -Threads 12
```

## Reproducible CPU evidence

One local one-thread benchmark on an MSI Modern 14 C12M (Intel Core i5-1235U,
Windows 11, Python 3.12.10, NumPy 2.1.3) used nine alternating runs after one
warm-up. It measures two narrow float32 cases; neither is a general framework
comparison.

| Workload | Kernelyra median | NumPy median | Result for this machine only |
|---|---:|---:|---|
| 1,000 identical logistic updates, 8,192 × 64 | 0.1337745 s | 0.1300291 s | NumPy `1.029×` faster; identical accuracy (99.0356%) |
| Copy → impute → normalize → clip, 8,192 × 64 | 0.0022132 s | 0.0023485 s | Kernelyra `1.061×` faster; 75 repaired values and equal checksum |

The native model and NumPy model have the same final accuracy in the training
case; the absolute final-loss difference is `1.49e-8`. A separate, untimed
pipeline probe observed all five native roles (`mask: 31`) in the random-batch
path. It is capability evidence and is excluded from both timing results.
These measurements do not prove performance on other hardware, datasets,
models, memory limits or workloads. See the [method and runner](benchmarks/cpu/README.md)
and [raw nine-run JSON](benchmarks/cpu/results/modern-14-c12m-v0.6.0a1.json).

<a id="limits"></a>

## Limits and roadmap boundary

The tested release target is **Windows x64 with Python 3.11–3.13**. Kernelyra
0.6 trains tabular models only. It does not include built-in trainers for LLMs,
images, audio, video, 3D, or other modalities. Recognizing an extension is not
the same as extracting it, training it, or supporting it as a model container.
`hybrid` requires a detected accelerator and an installed compatible optional
backend; it is not a promise that the bundled native backend trains on every GPU.

## More information

Run `python -m kernelyra --help` for commands, use
`python -m kernelyra formats` for capability levels, and see the
[changelog](CHANGELOG.md), [contribution guide](CONTRIBUTING.md),
[security policy](SECURITY.md), [third-party notices](THIRD_PARTY_NOTICES.md),
and [license](LICENSE).
