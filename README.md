<p align="center">
  <img src="assets/brand/kernelyra-logo.png" alt="Kernelyra" width="520">
</p>

<p align="center">
  <a href="https://github.com/Solveran601/Kernelyra/actions"><img src="https://img.shields.io/github/actions/workflow/status/Solveran601/Kernelyra/ci.yml?branch=main&label=CI" alt="CI status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0 license"></a>
  <img src="https://img.shields.io/badge/status-alpha-f59e0b" alt="Alpha status">
  <a href="README.ru.md">Русская версия</a>
</p>

<p align="center"><img src="assets/brand/kernelyra-mark-animated.svg" alt="Kernelyra mark" width="72"></p>

<p align="center"><strong>Native-first tabular model training with resource-aware safety.</strong></p>

Kernelyra 0.3.0a1 is a terminal-first library for training and fine-tuning
tabular models. It provides one automatic planning path through the CLI, Python
API, and a JSONL protocol used by the bundled language SDKs.

## What works today

- tabular binary classification, multiclass classification, and regression;
- direct training from CSV, TSV, JSONL/NDJSON, numeric NPZ, and Parquet;
- bounded-memory streaming for large compatible tabular files and folders;
- bundled native backend, plus NumPy and optional PyTorch or TensorFlow/Keras backends;
- automatic backend, hardware profile, batch size, and model-width planning;
- four execution profiles: weak PC, balanced PC, powerful PC, and workstation;
- checkpoints, resume, held-out evaluation, result-driven stopping, and Model Guard;
- CPU/RAM/GPU budgets, isolated workers, NaN guards, and an emergency stop;
- Python, CLI, and JSONL SDK adapters for C, C++, C#, Rust, Go, PHP, Java,
  Kotlin, Swift, and Ruby.

## Install from source

There is no PyPI package or GitHub Release yet. Install the current alpha from
a source checkout:

```powershell
git clone https://github.com/Solveran601/Kernelyra.git
Set-Location Kernelyra
python -m pip install -e .
```

Use `.[data]` only when Parquet support is needed, `.[torch]` for PyTorch, and
`.[tensorflow]` for TensorFlow/Keras.

## Quick start

```powershell
python -m kernelyra doctor
python -m kernelyra plan .\data\train.csv --target label
python -m kernelyra train .\data\train.csv --target label
```

Kernelyra inspects the dataset before training and reports the resolved plan.
If the target is ambiguous or a requested resource setting is unsafe, it stops
with an actionable error instead of guessing.

## Python

```python
from kernelyra import fit

result = fit("train.csv", "label", workspace="./project")
print(result.checkpoint)
```

For explicit control, use `Config` and `Engine` to choose a backend, profile,
resource budget, architecture, or quality guard.

## Current limits

Kernelyra 0.3 trains tabular models only. It does not include built-in trainers
for text, images, audio, video, 3D, or other multimodal data. File recognition
is not a claim that a format can be extracted or trained.

The supported release target is Windows x64 on Python 3.11–3.13. Linux, macOS,
and Windows ARM are not release targets for 0.3.

## More information

Run `python -m kernelyra --help` to see the available commands. The repository
keeps the runtime interface intentionally small during the 0.3 alpha period.
See the [license](LICENSE) for distribution terms.

Performance claims belong in reproducible reports that identify the workload,
hardware, package versions, accuracy, and memory measurement. Kernelyra ships a
benchmark harness, but it does not present unverified universal speed claims.
