# Datasets

Kernelyra 0.3 directly trains tabular data from CSV, TSV, flat JSONL/NDJSON,
numeric NPZ containing `x[rows, features]` and `y[rows]`, and Parquet
(`.[data]`/`.[parquet]`).

`kernelyra formats` can inspect a path and reports recognition, extraction, and
direct-training capability separately. Recognition alone does not mean that a
file can be extracted or trained. Built-in multimodal trainers are not part of
the 0.3 release.

Files up to the configured import limit are normalized into the workspace.
Larger CSV/TSV/JSONL/Parquet files and whole compatible folders use bounded-memory external streaming:

- one bounded scan validates schema, target and statistics;
- numeric features are standardized and categorical features are feature-hashed;
- train rows are read on demand with optional worker threads and prefetch;
- validation and test samples are bounded to 4096 rows each;
- checkpoints include the exact delivered-row cursor;
- a fingerprint rejects a changed source during resume.

External source files are never deleted when their Kernelyra metadata is removed.
NPZ currently has a 2 GiB safe expansion limit and does not claim a streaming
path.
