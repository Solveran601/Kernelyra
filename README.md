<p align="center"><img src="assets/brand/kernelyra-wave-top.svg" alt="" width="100%"></p>

<p align="center">
  <img src="assets/brand/kernelyra-logo.png" alt="Kernelyra" width="520">
</p>

<p align="center">
  <a href="https://github.com/Solveran601/Kernelyra/actions"><img src="https://img.shields.io/github/actions/workflow/status/Solveran601/Kernelyra/ci.yml?branch=main&label=CI" alt="CI status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0 license"></a>
  <img src="https://img.shields.io/badge/status-beta-ff9417" alt="Beta status">
  <a href="README.ru.md">Русская версия</a>
</p>

<p align="center"><img src="assets/brand/kernelyra-mark-animated.svg" alt="Animated Kernelyra mark" width="72"></p>

<p align="center"><strong>Native-first, resource-aware training for tabular data.</strong></p>

Kernelyra **0.7.0b1 (beta)** is a local terminal-first library for
tabular model training. The same planning path is available through the CLI,
Python API, PowerShell module, and JSONL protocol used by the bundled SDKs.

<p align="center">
  <a href="#capabilities">Capabilities</a> ·
  <a href="#install">Install</a> ·
  <a href="#powershell">PowerShell</a> ·
  <a href="#limits">Limits</a>
</p>

<a id="capabilities"></a>

## What works in this beta

- Binary and multiclass classification, plus regression, on tabular data.
- Direct training from CSV, TSV, JSONL/NDJSON, numeric NPZ, and optional
  Parquet input.
- Bundled native and NumPy CPU backends; optional PyTorch and TensorFlow/Keras
  backends provide the strict `hybrid` (CPU + GPU) path when a usable GPU is
  detected. The native core has five observable roles: a C ABI
  with checked-size guards, C++ dispatch, Rust policy, Fortran numeric kernels,
  and Zig memory kernels.
  A model's `native_execution` trace reports engines that actually participated
  in its native calls; it never guesses that every engine ran.
- Explicit `cpu` or `hybrid` execution and developer-set CPU/RAM/GPU/thread
  limits. A single automatic policy resolves bounded thread, bulk-dispatch,
  chunk, prefetch, and arena settings; it does not classify the user's PC.
  Persistent checkpoints are **off by default**. Users explicitly select
  `none`, `best`, or `last` for resume/final output; Model Guard can restore a
  model only when `checkpoint_rollback="best"` was explicitly requested.
- Data Doctor: bounded preflight findings, a signed dataset contract, a
  deterministic split recommendation, and a variable-range chunk plan.
- Split percentages, an explicit context/group column, and target/minimum/
  maximum variable-chunk sizes can be set through Python, CLI, PowerShell, or
  TOML. Streaming uses those exact values; a context key is excluded from
  learning features and never crosses a split.
- UTF-8 text preparation: native chunk planning when the native core is
  available, plus a reversible byte tokenizer and causal loss masks for a
  future trainer. This is not an LLM trainer.
- Conversation-safe streaming preparation for `.txt`, `.md`, `.log`, JSONL/
  NDJSON, and Telegram `result.json`. It reads one message at a time, never
  splits a source message, and closes chunks on conversation changes, explicit
  time gaps, size limits, or a conservative lexical topic heuristic. It does
  not write prepared data unless the caller supplies an output path, and it is
  not a built-in LLM trainer.
- Model Guard V2: finite-metric checks plus saved score-trend evidence in the
  run health record. Its public evaluator is side-effect-free: it reports
  evidence and never silently creates, restores, or deletes a checkpoint.
- Portable JSON/HTML experiment reports and explicit CPU/RAM/GPU budgets.
- A deterministic `kernelyra native self-test` checks the installed native ABI
  with a few in-memory float32 operations. It neither creates a workspace nor
  trains a model.

Data Doctor is intentionally bounded: its findings describe the inspected
sample, not every row in a dataset. For materialized training, classification
uses deterministic stratification and a detected time-like column preserves
the supplied input order. For CSV, TSV, JSONL/NDJSON, and Parquet, AutoTrainer
routes a detected group/context column through the streaming splitter: each
context stays in one split and that identifier is excluded from features.
Formats without that streaming path still receive an explicit leakage warning.

<a id="install"></a>

<p align="center"><img src="assets/brand/kernelyra-matrix-flow.gif" alt="Animated matrix flow" width="560"></p>

<p align="center"><sub>Native bulk updates reuse bounded buffers while the final step reports the loss.</sub></p>

## Install

This beta is published as a Windows x64 wheel in
[GitHub Releases](https://github.com/Solveran601/Kernelyra/releases/tag/v0.7.0b1).
PyPI publication is not enabled. Install the release wheel with Python 3.11–3.13:

```powershell
python -m pip install "https://github.com/Solveran601/Kernelyra/releases/download/v0.7.0b1/kernelyra_ai-0.7.0b1-py3-none-win_amd64.whl"
```

Or install from a source checkout:

```powershell
git clone https://github.com/Solveran601/Kernelyra.git
Set-Location Kernelyra
py -3.13 -m pip install .
```

Install `.[data]` for Parquet, `.[torch]` for PyTorch, or `.[tensorflow]` for
TensorFlow/Keras.

<a id="powershell"></a>

## CLI from PowerShell

```powershell
python -m kernelyra version
python -m kernelyra doctor
python -m kernelyra execution
python -m kernelyra --json native self-test
python -m kernelyra plan .\data\train.csv --workspace .\runs --target label --execution cpu --cpu 100 --ram 85 --threads 12
```

Before training, inspect bounded data-health evidence explicitly when useful:

```powershell
python -m kernelyra dataset doctor .\data\train.csv --workspace .\runs --target label
```

Preview the automatic execution policy before allocating a training run:

```powershell
python -m kernelyra tune --execution cpu --records 100000 --features 64 --batch-size 128
```

Allocation bounds, context boundaries, and Model Guard are safety invariants.
For a durable best model and automatic rollback, opt in explicitly with
`--checkpoint-final best --checkpoint-rollback best`. Without that choice,
Model Guard stops a degrading run but does not create or restore a checkpoint.

Prepare a large Telegram export without loading it into memory or writing an
output file:

```powershell
python -m kernelyra text plan .\Telegram\result.json --maximum-characters 8192
```

Add `--output .\prepared\telegram-chunks.jsonl` only when you want the
prepared JSONL file written.

## Python

```python
from kernelyra import Engine, ModelGuard, iter_conversation_chunks, native_core_self_test

with Engine(workspace="./runs") as kernelyra:
    health = kernelyra.doctor("train.csv", "label")
    result = kernelyra.fit("train.csv", "label", backend="auto")
    run = kernelyra.run(result.run.id)
    print(run.status, run.metrics, run.logs(limit=20))
    report = run.report("./run-report.json")  # writes only because a path was supplied

print(report["output"])

guard = ModelGuard(degradation_margin=.03, degradation_patience=3)
evidence = guard.inspect(
    score=.71, loss=.42, metrics={"accuracy": .71}, best_score=.75,
    scores=[.75, .73, .71],
)
print(evidence["status"], evidence["restore_recommended"])

native = native_core_self_test()
if native["available"] and not native["ok"]:
    raise RuntimeError(native["diagnostic"])

for chunk in iter_conversation_chunks("Telegram/result.json"):
    print(chunk.boundary, len(chunk.messages), chunk.text[:80])
```

<p align="center"><img src="assets/brand/kernelyra-training-signal.gif" alt="Animated neural-network training signal" width="560"></p>

<p align="center"><sub>Native computation is guarded at every update; the diagram is illustrative, not a claim of an LLM trainer.</sub></p>

Inspect settings before opening a dataset or workspace:

```python
from kernelyra import Settings

settings = Settings().cpu_only().resources(cpu=85, ram=70, threads=8)
print(settings.explain())  # no filesystem writes
```

Kernelyra is an ordinary installed Python library: it does not create or
activate a virtual environment. `import kernelyra` has no filesystem side
effects. Stateful API calls require `workspace=...`; this explicit directory
is the only place where Kernelyra creates a run database, copied tabular data,
or checkpoints. A text chunk iterator has no write side effect.

For PowerShell command names, import the bundled module from a checkout:

```powershell
Import-Module .\powershell\Kernelyra.psd1 -Force
Test-KernelyraDataset .\data\train.csv -Target label -Workspace .\runs
Get-KernelyraDataContract .\data\train.csv -Target label -Workspace .\runs
Get-KernelyraNativeStatus
Get-KernelyraCpuTuning -Records 100000 -Features 32 -BatchSize 64
Get-KernelyraCpuTuning -Records 100000 -Features 64 -BatchSize 128
Get-KernelyraPlan .\data\train.csv -Target label -Workspace .\runs
Start-KernelyraTraining .\data\train.csv -Target label -Workspace .\runs -Execution cpu -Cpu 100 -Ram 85 -Threads 12
```

<a id="limits"></a>

## Limits and roadmap boundary

The tested release target is **Windows x64 with Python 3.11–3.13**. Kernelyra
0.7 trains tabular models only. It does not include built-in trainers for LLMs,
images, audio, video, 3D, or other modalities. Recognizing an extension is not
the same as extracting it, training it, or supporting it as a model container.
`hybrid` requires a detected accelerator and an installed compatible optional
backend. Native and NumPy are deliberately CPU-only; a forced hybrid run never
silently falls back to them.

## More information

Run `python -m kernelyra --help` for commands, use
`python -m kernelyra formats` for capability levels, and see the
[changelog](CHANGELOG.md), [contribution guide](CONTRIBUTING.md),
[security policy](SECURITY.md), [third-party notices](THIRD_PARTY_NOTICES.md),
and [license](LICENSE).

<p align="center"><img src="assets/brand/kernelyra-wave-bottom.svg" alt="" width="100%"></p>
