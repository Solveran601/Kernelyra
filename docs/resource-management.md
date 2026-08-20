# Resource management

Profiles are `low-memory`, `balanced`, `performance`, `workstation` and `custom`;
`eco` remains an SDK compatibility alias. Workstation requires at least 32 CPU
threads, 64 GB RAM and 16 GB GPU VRAM, or an exceptional 64-thread/128-GB
CPU-only machine. Detection reads CPU/RAM and a bounded `nvidia-smi`
query only—no startup benchmark or heavy ML import.

Every run reports requested limits separately from scheduler enforcement,
OS enforcement, backend enforcement, unsupported controls and degraded
fallbacks. The supported Windows release uses a Job Object for process-tree
termination and best-effort memory/CPU caps. TensorFlow configures GPU
memory growth or a requested logical-device limit. GPU control is not claimed
for a backend that cannot enforce it.

The repository retains inactive POSIX resource-control groundwork for future
ports, but Linux and macOS are not part of the 0.3 support or release matrix.

Auto batch uses dataset rows/features, RAM and profile, stays within a reported
safe range and adapts only from validation stability or OOM. A risky manual batch
requires explicit acceptance. Step count is an emergency ceiling, not the goal.

Model Guard validates on a configurable interval, retains the best accepted
checkpoint and rolls back after a configurable number of consecutive score
drops. `degradation_margin`, `degradation_patience`, `min_improvement`,
`early_stopping_patience` and `target_patience` are available in Python, TOML,
environment variables, CLI and the language-neutral protocol. The guard cannot
promise that a model will never degrade on unknown real-world data; it prevents
delivery of a checkpoint that is measurably worse on the configured validation
and test splits.

`custom` is a complete manual profile. It accepts explicit CPU/RAM/GPU limits,
batch size, backend, architecture, precision, model widths, optimizer settings
and every Model Guard threshold. Omitted values retain balanced safe defaults.
