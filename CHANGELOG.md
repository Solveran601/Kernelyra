# Changelog

## Unreleased

- Added an active C core under `native/core/c`: overflow-safe matrix and
  random-minibatch contracts are checked before any native buffer resize or
  pointer arithmetic.
- Added a Rust batch planner and a Zig one-pass feature-plus-target gather to
  the random-minibatch path. The previous C++ sampler/gather remains a checked
  fallback when a component is unavailable.
- Strengthened the Fortran pre-update guard: it now proves every candidate
  weight and bias remains representable as finite float32 before parameters are
  mutated. The native contract test covers this no-mutation overflow case.
- Made the source `native build` command compile the C core too, matching the
  CMake build graph.

## 0.6.0a1 — 2026-08-28

- Added a fused Zig preprocessing route for the public `impute → normalize →
  clip` operation. It performs the compatible float32 transform in one native
  traversal when all three options are requested, while the checked fallback
  remains available for every other flag combination.
- Added the Rust deterministic minibatch sampler to the native random-batch
  path and retained the C++ sampler as a checked fallback. Zig gathers the
  selected rows and the existing Fortran dense kernels execute the portable
  numeric update; the C ABI/C++ dispatcher remains the ownership boundary.
- Added a per-model `native_execution` trace. It records native engines that
  actually ran and never represents a compiled or enabled component as proof
  of execution.
- Strengthened the wide Fortran gradient reduction with OpenMP SIMD and
  double-precision accumulation before the float32 gradient writeback.
- Expanded the maintained CPU evidence into two comparable workloads: dense
  binary training and fused preprocessing. The raw JSON includes all runs,
  runtime metadata, correctness checks, and an explicitly untimed five-engine
  capability probe. No universal performance claim is made.

## 0.5.0a3 — 2026-08-27

- Added immutable built-in algorithm packs plus validated user packs that can
  be cloned and edited through Python, CLI or PowerShell. Pack algorithms
  materially control native threads, bulk dispatch, bounded prefetch, context
  chunk targets and reusable arena ceilings; correctness guards stay mandatory.
- Added a repeated full-batch native ABI, reused thread-local gradient
  workspaces, runtime AVX2/FMA dispatch and workload-gated OpenMP execution.
- Expanded the active Fortran training guard/workspace kernels and Zig bounded
  arena/batch planning used by the native CSV streaming path.
- Added one maintained CPU benchmark runner and raw local evidence with exact
  hardware/runtime metadata. The README reports only that measured case and
  does not generalize it to other hardware, datasets or frameworks.

## 0.5.0a2 — 2026-08-26

- Replaced public PC-strength modes with explicit `cpu`/`hybrid` execution,
  developer-controlled CPU/RAM/GPU/thread limits, and four algorithm packs.
- Added composable PowerShell commands for execution inspection, chunk plans,
  direct command invocation, and pipeline-friendly planning/training.
- Kept binary-classification hot loops inside the C++ AVX2 path and removed
  redundant cross-entropy logarithms for standard 0/1 labels. The matching
  CPU benchmark remains reproducible and does not claim a NumPy win.

## 0.5.0a1 — 2026-08-26

- Added Data Doctor: bounded sample findings, signed dataset contracts,
  deterministic split recommendations and variable-range chunk plans.
- Added deterministic stratified and source-order temporal materialized splits.
  Group/context detection is explicitly advisory until a group-exclusive
  splitter is implemented.
- Added Model Guard V2 score-trend evidence, portable experiment reports and
  `dataset doctor` / `report` CLI commands.
- Added the bundled PowerShell module and a source-checked V3 workflow
  benchmark that records local measurements without universal speed claims.
- Published a reproducible CPU matrix for Kernelyra plus ten ML libraries,
  with an independent hold-out score, exact versions and a separate
  non-comparable tree/online section.
- Updated the public README files to state only tabular capabilities and
  tested Windows x64/Python 3.11–3.13 support.

## 0.4.0a1 — 2026-08-21

- Restricted the 0.3 release and CI support matrix to Windows x64; other OS
  groundwork remains experimental and is not advertised as supported.
- Expanded the easy Python API with reusable defaults, configuration
  copy/merge/reset, hardware/stopping helpers, inspection and sequential
  multi-dataset planning/training.
- Reworked the native component layout into two active parts: Zig owns explicit
  memory operations and Fortran owns dense training arithmetic. The C++ layer
  now remains the ABI, streaming and safe-fallback boundary.

- Removed Plugin Center, plugin SDK, entry-point discovery, browser assets and
  the reference extension package.
- Added a closed built-in catalogue of 535 format routes with separate
  recognition, extraction and direct-training levels.
- Added bounded-memory extraction for 201 text/code/structured formats and
  deterministic whole-folder extraction.
- Added streaming folder datasets with deterministic ordering, schema
  validation and immutable-source fingerprints.
- Added explicit architecture/model-format contracts; unsupported
  Transformer/GGUF training fails clearly instead of being simulated.

- Added a single C ABI linking C++17, Zig 0.16 and Fortran, with runtime
  feature masks and portable fallbacks.
- Added Zig 64-byte aligned allocation, copy/zero/normalization operations,
  Rust zero-copy native bindings and Fortran dot/AXPY/gradient/update kernels.
- Validated bounded-memory ingestion and full train-split passes at
  100/200/300 MiB and 1 GiB; disposable datasets and one-off harnesses are not
  retained in the source tree.

- Added a bundled C++17 training core with a stable C ABI, AVX2/FMA runtime
  dispatch, OpenMP thresholds, platform wheels and native/NumPy/PyTorch/TensorFlow
  benchmark modes.
- Added one-pass bounded-memory native numeric CSV/TSV schema analysis and
  checkpoint-restorable train/validation/test streams for datasets larger than RAM.
- Added workstation 100% CPU/GPU scheduler capacity with a 5%/OS RAM reserve,
  chunked worker training, and model-degradation rollback to the best checkpoint.
- Moved heavy frameworks out of `requirements.txt`; PyTorch, TensorFlow, pandas,
  PyArrow, gateway and MCP packages remain explicit optional extras.
- Added a loopback-only Plugin Center with one-time browser sessions, isolated
  validation and workspace-scoped enable/disable state; training remains CLI/library-only.
- Added a fourth `workstation` hardware and benchmark profile.

## 0.3.0a1 — 2026-08-12

- Removed the browser UI, JavaScript/TypeScript assets and desktop launcher; direct terminal commands are now the primary product surface.
- Added `AutoTrainer`, top-level `plan`/`train`/`finetune`, TOML/environment/explicit configuration precedence and exact manual-setting validation.
- Added a lazy real PyTorch backend with safe weights-only fine-tuning, CUDA limits, mixed precision and gradient clipping.
- Added bounded-memory CSV/TSV/JSONL/Parquet streaming with source fingerprints, parallel encoding, prefetch and checkpoint cursor restoration.
- Added the stable `kernelyra-jsonl/1` stdio contract and adapters for C, C++, C#, Rust, Go, PHP, Java, Kotlin, Swift and Ruby.
- Added the shared easy-library API (`Config` plus `fit`/`tune`) for Python, Go, C++17, Rust and C#, including CMake, Cargo, Go module and NuGet package metadata.
- Added a reproducible direct-framework benchmark for wall time, process-tree RAM, held-out score and checkpoint size without making unsupported speed claims.
- Split small core, backend, data, gateway, MCP and full optional dependency sets.

## 0.2.0rc1 — 2026-08-01

- Added production dataset ingestion for CSV, JSONL, NPZ and optional Parquet with typed manifests, content-addressed storage, decompression limits and leakage-aware preprocessing.
- Added result-driven binary classification, multiclass classification and regression on NumPy and TensorFlow, including train/validation/test separation, adaptive batches, atomic checkpoints and compatible resume.
- Moved training into authenticated spawned workers with heartbeat, timeout, crash containment and platform resource enforcement reporting.
- Expanded the typed sync/async SDK, CLI, `/api/v1`, SSE dashboard and MCP surface for dataset, run, metrics, logs, export, approvals and maintenance workflows.
- Added expiring scoped agent sessions, one-time approvals, CSRF, Host/Origin validation, rate limits, redacted exports and secret-free workspace recovery manifests.
- Added a versioned plugin SDK, isolated plugin discovery, diagnostics and a separately buildable reference backend plugin.
- Replaced static-file launch behavior with a protected local desktop launch flow and four hardware profiles.
- Added reproducible wheel, sdist and source-bundle tooling, semantic OpenAPI checks, security/failure tests, native probe protocol and multi-OS release/soak CI gates.

## 0.2.0a6 — 2026-08-01

- Replaced direct agent-secret access with expiring agent sessions scoped to roots, actions, workspace and client ID.
- Enforced MCP filesystem roots inside the daemon, including raw HTTP path inspection and imports.
- Added deterministic source-ZIP construction with extraction and clean-content verification.
- Strengthened clean-source tests to reject Python caches, secrets, PID/lock files, arrays and runtime data.
- Replaced the generator-sensitive OpenAPI document hash with a semantic `/api/v1` contract snapshot.
- Added pinned release constraints plus minimum/latest-compatible dependency CI checks.

## 0.2.0a5 — 2026-08-01

- Added a separate least-privilege agent credential for every `/api/v1/mcp/*` route.
- Protected user state, run, dataset, log, hardware and path-inspection reads; only minimal health remains public.
- Required a one-time UI session before serving the dashboard or its state.
- Made the PowerShell launcher establish the same protected browser session as the BAT launcher.
- Added security E2E coverage and an automatic `/api/v1` OpenAPI contract snapshot.

## 0.2.0a4 — 2026-08-01

- Required protected local user or browser sessions for ordinary state-changing HTTP routes.
- Kept MCP on dedicated agent routes with allowlists and one-time approvals.
- Resolved relative MCP tool paths independently of process working directory.
- Removed network API tokens from background daemon argv and normalized IPv6 URLs.
- Included source launchers in sdist and froze the `/api/v1` shape for the 0.2 RC soak cycle.

## 0.2.0a3 — 2026-08-01

- Made worker lifecycle outcomes explicit and finalized terminal states only after backend release.
- Fixed the pause-to-stop race during a blocking backend close and protected transitions from stale progress writes.
- Resolved relative MCP allowlist roots against the configuration directory/workspace.
- Detached background daemons from POSIX terminal sessions and strengthened installed-wheel lifecycle verification with a slow backend plugin.

## 0.2.0a2 — 2026-08-01

- Added acknowledged `pausing` and `stopping` transitions with live-worker resource accounting.
- Guaranteed backend worker cleanup for completion, pause, stop and error exits.
- Removed machine-specific paths and added portable Windows launchers plus a neutral configuration example.
- Replaced the abbreviated license with the complete Apache License 2.0 text.
- Added installed-wheel daemon, UI, CLI lifecycle and MCP stdio release verification.
- Isolated third-party plugin failures and exposed structured plugin diagnostics.
- Disabled the built-in web UI in authenticated non-loopback mode.
- Bounded directory inspection without materializing more than 5,000 files.
- Added friendly optional-MCP installation guidance and stricter source-artifact checks.

## 0.2.0a1 — 2026-08-01

- Added inert installable Python package and typed public SDK.
- Split storage, ingestion, batch planning, runtime, backends, server, CLI and MCP.
- Added SQLite schema migrations and legacy 0.1 history import.
- Replaced `http.server` and `cgi.FieldStorage` with FastAPI and `UploadFile`.
- Added versioned `/api/v1` endpoints and OpenAPI.
- Added backend/ingestor discovery through Python entry points.
- Made TensorFlow fully lazy and removed import-time scheduler/runtime creation.
- Added stdio MCP with workspace allowlist, one-time user approvals and action audit log.
- Added a single daemon control plane, protected non-loopback mode, portable native probe discovery and packaged UI.
- Added result-driven adaptive batch and deterministic seed configuration.
- Added Windows-safe SQLite connection closure and automated tests.
