# Migration from 0.2 alpha to the 1.0 contract

Back up the workspace, stop its daemon, then run `kernelyra migrate`. The command
creates a SQLite backup before applying idempotent schema migrations and reports
integrity. User datasets/checkpoints are not deleted.

Public `/api/v1` and existing SDK names remain. New run fields have defaults.
Training now uses a spawned worker, explicit task type and train/validation/test
split. Resource percentages no longer imply OS enforcement; inspect the per-run
enforcement object. Core `eco` maps to `low-memory` but remains accepted.

Dynamic plugin backends and Plugin Center are removed; use a reviewed built-in
backend. MCP clients must exchange the
agent bootstrap credential for a scoped session and obtain approvals for import,
start, resume and export.
