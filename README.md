<p align="center">
  <img src="assets/brand/kernelyra-logo.png" alt="Kernelyra" width="520">
</p>

<p align="center">
  <a href="https://github.com/Solveran601/Kernelyra/actions"><img src="https://img.shields.io/github/actions/workflow/status/Solveran601/Kernelyra/ci.yml?branch=main&label=CI" alt="CI status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0 license"></a>
  <img src="https://img.shields.io/badge/runtime-Python%203.11%2B-3776ab" alt="Python 3.11 or newer">
</p>

<p align="center"><img src="assets/brand/kernelyra-mark-animated.svg" alt="Kernelyra mark" width="72"></p>

<p align="center"><strong>Native-first training with resource-aware safety.</strong></p>

Kernelyra is a terminal-first, code-first engine for training and fine-tuning
tabular AI models. Training has no browser controls and the extension/plugin
system has been removed. The same automatic planner is available from the CLI,
Python API and a stable JSONL protocol used
by the bundled multi-language SDKs.

## What works

- bundled native backend (C++17 ABI/streaming + Zig memory + Fortran training) plus optional PyTorch, TensorFlow/Keras and NumPy backends;
- binary classification, multiclass classification and regression;
- CSV, TSV, JSONL/NDJSON, numeric NPZ and Parquet ingestion;
- streaming training from whole folders of compatible tabular files;
- 535 built-in format routes across text/code, documents, tables, images,
  audio, video, 3D, point clouds, geodata, archives and model containers;
- real bounded-memory text extraction for 201 text/code/structured formats,
  including deterministic whole-folder traversal;
- explicit `architecture + model_format` validation without pretending that
  recognition is the same as extraction or training;
- C++ bounded-memory numeric CSV/TSV streaming and general bounded-memory JSONL/Parquet/mixed-table streaming;
- automatic hardware profile, backend, task, batch size and network widths;
- four execution programs inside one API: weak PC, balanced PC, powerful PC and workstation;
- exact precedence: Python/CLI option > environment > TOML > automatic value;
- adaptive batches with an explicit warning/confirmation for unsafe manual sizes;
- result-driven stopping, best/last atomic checkpoints, pause, resume and final test evaluation;
- configurable Model Guard (validation interval, score margin/patience, plateau patience and target stability) that delivers and tests the best accepted checkpoint;
- CPU/RAM/GPU policy, background worker isolation, NaN guards and emergency Ctrl+C stop;
- Python plus C, C++, C#, Rust, Go, PHP, Java, Kotlin, Swift and Ruby adapters.

The built-in router reports three separate capability levels: recognized,
extractable and directly trainable. Core training currently supports the tabular
formats above; multimodal trainers are not claimed until they are implemented.

## Быстрый старт в PowerShell

После установки код писать не нужно: путь к датасету — единственный обязательный
аргумент. Kernelyra сам определит поддерживаемый табличный формат, цель и
безопасный план; при неоднозначной цели он остановится с понятным запросом, а не
будет угадывать обучение.

```powershell
# После публикации в PyPI (0.3 пока является pre-release):
python -m pip install --pre "kernelyra-ai[data]"
python -m kernelyra train C:\data\train.csv
python -m kernelyra finetune C:\models\best.npz C:\data\new_rows.csv
python -m kernelyra plan C:\data\folder
python -m kernelyra native status --json
```

После публикации пакета те же команды работают короче: `kernelyra train
C:\data\train.csv`. Для тонкой настройки добавляй только нужные параметры,
например `--backend native --profile low-memory` или `--max-steps 500`; все
остальные значения остаются автоматическими.

## Four execution programs

`low-memory` (and legacy `eco`) resolves to the weak-PC program: native CPU,
zero prefetch and strict streaming. `balanced` raises safe parallelism;
`performance` and `workstation` use progressively larger queues, workers and
batch ceilings, and prefer an installed GPU backend when an accelerator is
available. All four fall back safely to the native CPU backend.

NVIDIA discovery is automatic and lightweight. For a non-NVIDIA accelerator
supported by the installed PyTorch/TensorFlow runtime, set one hint before
starting Kernelyra, for example `$env:KERNELYRA_ACCELERATOR = "rocm"`, `metal`
or `directml`; device probing still happens inside the isolated run worker.

## Install

```console
# Из исходного checkout — режим разработки:
python -m pip install -e ".[torch,data]"
kernelyra native build
kernelyra doctor --json
kernelyra formats --json
```

Windows wheels include the native core, so end users do not need a compiler.
Release builds use Zig for aligned allocation/copy/zero/normalization and
Fortran for dense training arithmetic (dot, AXPY, gradients and weight updates).
The C++ layer supplies the ABI, data stream and safe fallback. A source checkout
needs Zig and gfortran once, then builds with `kernelyra native build`. PyTorch,
TensorFlow, pandas and PyArrow are optional; install `.[torch]`, `.[tensorflow]`,
`.[data]` or `.[full]` only for the workloads that need them.

## Terminal workflow

```console
kernelyra --workspace ./project plan ./data/train.csv --target label
kernelyra --workspace ./project train ./data/train.csv --target label
kernelyra --workspace ./project train ./data/train.csv --backend native
kernelyra --workspace ./project train ./data/train.csv --backend torch --batch-size 64
kernelyra --workspace ./project train ./data/folder --backend torch --architecture mlp --model-format kernelyra-npz
kernelyra --workspace ./project finetune ./model.pth ./data/train.csv --target label
kernelyra --workspace ./project infer RUN_ID --requests 200 --json
```

`start_worker.bat`, `start_worker.ps1` and `python worker.py` forward arguments to
the same terminal CLI. They do not open a browser.

Manual batch values outside the safe range are rejected until
`--accept-batch-risk` is supplied. Training ends when the target is reached
stably, progress plateaus, the user stops it, or the emergency step limit is hit;
epochs are not the product-level stopping contract.

The engineering performance objective is at least 2x better throughput or
resource efficiency than matched framework baselines where profiling proves it.
It is not a universal claim: every published number must be re-measured on the
named hardware class with a disposable, source-controlled measurement harness.

Measured on the development i5-1235U/8 GB Windows laptop, reproducible local
benchmarks currently show:

- native float32 binary kernel: 2.18–2.81x the matching NumPy kernel;
- numeric CSV decode + standardization: 3.26x NumPy and 2.14x pandas, with
  median peak memory 91 MB vs 267 MB and 118 MB;
- bounded-memory Kernelyra numeric CSV batches: 49.37x the previous internal
  Python/pandas streaming path, with 47 MB vs 170 MB peak memory.
- generated 100/200/300 MiB and 1 GiB numeric CSV files stay at about 47-48 MB
  peak RAM. Steady-state native ingestion was 18.85-57.38x the old internal
  Python stream in recorded runs. A full 1 GiB train-split pass was 1.49x the
  matched NumPy loop; 100-300 MiB runs were near parity (0.95-1.01x).

These are scoped results, not a universal TensorFlow/PyTorch superiority claim.
The direct small-workload baseline is still faster in NumPy, and deep-network
parity remains future engine work. Every published claim must include workload,
hardware, score and memory evidence.

## Python library

```python
from kernelyra import fit

# Everything is automatic. Dataset and target are enough.
result = fit("train.csv", "label", workspace="./project")
print(result.checkpoint)
```

Every option is still available without building a low-level run object:

```python
from kernelyra import Config, Engine

settings = (
    Config()
    .backend("native")
    .architecture("linear")
    .model_format("kernelyra-npz")
    .goal(0.95)
    .steps(20_000)
    .batch(64, accept_risk=True)
    .resources(cpu=70, ram=60, gpu=75)
    .optimizer(learning_rate=0.0003, weight_decay=0.01)
    .model(256, 128, 64, precision="auto")
    .data(workers=4, prefetch=2)
    .quality(evaluation_interval=100, min_improvement=0.0005,
             early_stopping_patience=18, target_patience=3)
    .guard(margin=0.03, patience=3)
)

with Engine("./project", settings=settings) as engine:
    info = engine.inspect("train.csv")
    plan = engine.plan("train.csv", "label", settings=settings)
    result = engine.fit("train.csv", "label", settings=settings)
```

Configuration objects can be copied, merged with dictionaries or other
configs, and reset back to automatic resolution with `unset()`/`automatic()`.
`Engine.configure()` sets reusable defaults; `hardware` and `capabilities`
expose safe snapshots. `plan_many()` and `fit_many()` process lists of files or
folders sequentially so Windows resource limits remain enforceable.

The original `train`, `finetune`, `AutoTrainer` and low-level `Workspace` APIs
remain supported.

See [auto training](examples/auto_train.py), [fine-tuning](examples/finetune.py),
the [CLI reference](docs/cli.md), [architecture](docs/architecture.md),
[datasets](docs/datasets.md), [backends](docs/backends.md), and
[language protocol](sdks/README.md).

## Configuration

Place `kernelyra.toml` in the workspace or pass `--config`:

```toml
[training]
backend = "auto"
architecture = "auto"
model_format = "auto"
profile = "auto"
batch_size = 64
max_steps = 5000
target_metric = 0.93
precision = "auto"
data_workers = 4
prefetch = 2
evaluation_interval = 100
min_improvement = 0.0005
degradation_margin = 0.03
degradation_patience = 3
early_stopping_patience = 18
target_patience = 3
```

Every field can be overridden by CLI/Python. Environment names use the
`KERNELYRA_` prefix, for example `KERNELYRA_BACKEND`,
`KERNELYRA_ARCHITECTURE`, `KERNELYRA_MODEL_FORMAT` and
`KERNELYRA_BATCH_SIZE`; the complete list is in [CLI documentation](docs/cli.md).

## One engine, many languages

Run `kernelyra --workspace ./project rpc` for the `kernelyra-jsonl/1` protocol.
The adapters under `sdks/` call that one process, so batching, safeguards and
training behavior never drift between languages. C++ and Rust are also used for
the stable native ABI and bounded-memory data path; SQL is a data-source concern.

Python, Go, C++, Rust and C# expose the same easy concepts: `Config`, `fit`,
`tune`, `checkpoint`, `status` and raw plan/metrics. See
[multi-language examples](docs/sdk.md).

Rust additionally exposes a `native` Cargo feature for direct zero-copy calls to
the C ABI, including Zig aligned memory operations and Fortran dot/gradient
kernels. The released core enables its two native components by default; its
component mask remains available for controlled fallback verification.

## Performance claims

Kernelyra is an orchestration and data engine that can run PyTorch or TensorFlow;
it is not honest to claim that the wrapper is universally faster than the
framework it runs. Use an isolated, disposable harness before publishing
numbers. Measure wall time, training work, peak process-tree RAM, held-out
accuracy and checkpoint size for Kernelyra and matching direct-framework
baselines, then remove generated datasets, models and one-off harnesses from the
source tree.

## Reproducible 10-library benchmark

The normal package does **not** install heavyweight frameworks.  The isolated
benchmark stack downloads exactly ten training dependencies — PyTorch,
TensorFlow, JAX, Flax, Optax, scikit-learn, XGBoost, LightGBM, CatBoost and
River — only when this command is requested:

```powershell
# В checkout репозитория:
.\scripts\install_benchmark_stack.ps1
```

For an installed release, the equivalent dependency command is:

```powershell
python -m pip install --pre "kernelyra-ai[benchmark]"
```

It creates `.benchmarks\frameworks-venv` and writes the complete JSON report to
`.benchmarks\framework-matrix.json`.  To reuse an existing environment:

```powershell
.\.benchmarks\frameworks-venv\Scripts\python.exe scripts\benchmark_tabular_frameworks.py
```

The report separates results instead of inventing an unfair leaderboard:

| Group | Compared libraries | Meaning |
| --- | --- | --- |
| `matched_linear` | Kernelyra, NumPy, PyTorch, TensorFlow, JAX, Flax/Optax, scikit-learn | Identical synthetic full-batch logistic-regression task; wall-time values may be compared only within this group. |
| `tree_not_matched` | XGBoost, LightGBM, CatBoost | Native tree training capability, recorded separately from linear throughput. |
| `online_linear_not_matched` | River | Online row-by-row learner, recorded separately. |

No command silently uploads the report or dataset.  Treat a result as evidence
only when the JSON report identifies the hardware, package versions, workload,
accuracy and the comparison group.

## Release automation

Pushing a regular commit runs CI; it does not publish anything.  Pushing a
verified tag such as `v0.3.0a1` runs `.github/workflows/release.yml`, builds the
Windows wheel plus source archives, verifies them, attaches them to a GitHub
Release and publishes the checksums.  A separate manual workflow dispatch can
publish the wheel and sdist to PyPI after its protected `pypi` environment and
Trusted Publisher are configured.  This prevents accidental public releases.

Kernelyra 0.3 supports Windows x64 on Python 3.11-3.13. Linux, macOS and Windows
ARM are not release targets yet and are not claimed as supported.
