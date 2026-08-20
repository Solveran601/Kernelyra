# Kernelyra security policy

Report vulnerabilities privately to the project owner with the affected
version, minimal reproduction and impact. Never attach a private dataset,
workspace secret or checkpoint.

## Trust boundaries

The local user who owns a workspace is trusted. Terminal users and MCP agents
are less privileged. Dynamic third-party code extensions are not loaded;
backend, ingestor and format registries contain reviewed built-ins only.

User HTTP routes require the local user secret or an explicitly configured
network bearer. Only minimal health is public. Agent sessions are random,
short-lived and revocable; requests require an approved Host and CORS is off.

MCP uses a separate bootstrap secret to obtain a short-lived scoped session.
The daemon binds it to workspace, client ID, allowed actions and resolved roots.
Agents cannot create user approvals. Import, start, resume and export
consume a user-issued action/resource-bound approval exactly once. Sessions and
approvals can be revoked.

## Data and process isolation

Imports reject direct symlinks/reparse sources, enforce real paths and recheck
the SHA-256 while copying to content-addressed storage. Copies, runtime arrays,
manifests and checkpoints use pending files plus atomic replacement. Uploads are
limited to 50 MB and path imports to 512 MB. NPZ uses `allow_pickle=False` and
rejects unsafe expansion ratios; JSONL lines, CSV fields, native output and IPC
frames are bounded. Files are parsed as data, never executed.

Each run uses a spawned process with authenticated versioned loopback IPC,
heartbeat, bounded messages and process-tree termination. A worker crash moves
only that run to a recoverable state and does not stop the daemon.

## Secrets and releases

Secrets use `secrets.token_urlsafe`, exclusive atomic creation and POSIX 0600;
Windows permissions are restricted on a best-effort basis by the user profile.
Secrets are never accepted in URL query strings, command arguments, logs,
OpenAPI examples, exported manifests or artifacts. Treat any secret found in an
external archive as compromised and rotate it.

Source ZIPs are built from an allowlist and scanned after extraction. Wheel,
sdist and source ZIP verification rejects runtime state, SQLite, locks, PID,
checkpoints, arrays, caches, logs, executables and personal absolute paths.

The detailed threat analysis and residual risks are in
[`docs/threat-model.md`](docs/threat-model.md).
