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

Kernelyra **0.5.0a2 (V3 alpha)** is a local terminal-first library for
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
  backends when installed.
- Explicit `cpu` or `hybrid` execution, developer-set CPU/RAM/GPU/thread
  limits, and four optional algorithm packs (`careful`, `balanced`,
  `throughput`, `maximum`). Packs alter bounded chunk/prefetch/working-set
  defaults; they do not classify the user's PC. Checkpoints, resume,
  held-out evaluation, and best-checkpoint restoration are included.
- Data Doctor: bounded preflight findings, a signed dataset contract, a
  deterministic split recommendation, and a variable-range chunk plan.
- Model Guard V2: finite-metric checks plus saved score-trend evidence in the
  run health record.
- Portable JSON/HTML experiment reports and explicit CPU/RAM/GPU budgets.

Data Doctor is intentionally bounded: its findings describe the inspected
sample, not every row in a dataset. For materialized training, classification
uses deterministic stratification and a detected time-like column preserves
the supplied input order. A detected group/context column is currently an
**advisory warning**; group-exclusive splitting is not implemented in 0.5.

<a id="install"></a>

## Install from source

PyPI publication is not configured. After the tagged GitHub Actions build succeeds,
Windows release artifacts for this alpha are attached to its GitHub pre-release.

```powershell
git clone https://github.com/Solveran601/Kernelyra.git
Set-Location Kernelyra
python -m pip install -e .
```

Install `.[data]` for Parquet, `.[torch]` for PyTorch, or `.[tensorflow]` for
TensorFlow/Keras.

<a id="powershell"></a>

## Three PowerShell commands

```powershell
python -m kernelyra doctor
python -m kernelyra execution
python -m kernelyra plan .\data\train.csv --target label --execution cpu --pack throughput --cpu 100 --ram 85 --threads 12
```

Before training, inspect bounded data-health evidence explicitly when useful:

```powershell
python -m kernelyra dataset doctor .\data\train.csv --target label
```

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
Get-KernelyraPlan .\data\train.csv -Target label
Start-KernelyraTraining .\data\train.csv -Target label -Execution cpu -Pack throughput -Cpu 100 -Ram 85 -Threads 12
```

<a id="limits"></a>

## Limits and roadmap boundary

The tested release target is **Windows x64 with Python 3.11–3.13**. Kernelyra
0.5 trains tabular models only. It does not include built-in trainers for LLMs,
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
