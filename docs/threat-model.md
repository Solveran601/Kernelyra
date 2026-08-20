# Threat model

Assets are datasets, checkpoints, run history, local compute and credentials.
Attack surfaces are local/network HTTP, MCP agents, filesystem
paths, uploads/archives, native probe output, worker IPC and SQLite.

Controls include loopback default, bearer requirement for network bind, Host
allowlists, CORS-off behavior, rate limits, random expiring/revocable tokens,
strict agent/user separation,
daemon-side realpath/action enforcement and one-time approvals. File controls
cover traversal/symlink/reparse rejection, atomic copy plus hash recheck, upload
and archive limits, no pickle, safe model loading and secret-free exports.

Workers use authenticated protocol-versioned bounded IPC, heartbeat and process
tree cleanup. SQLite uses WAL, foreign keys, busy timeout, schema versions,
transactions, integrity check and explicit repair. Release allowlists prevent
ZIP Slip inputs and runtime state packaging.

Residual risks: the same OS account can read its workspace; Windows ACL
hardening is best-effort; cgroup/GPU controls depend
on host/backend permissions; TLS is external in network mode. Kernelyra reports
these as unsupported/degraded rather than promising enforcement.
