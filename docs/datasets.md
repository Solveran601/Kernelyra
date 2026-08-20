# Datasets

Core trainable formats are CSV, TSV, flat JSONL/NDJSON, numeric NPZ containing
`x[rows, features]` and `y[rows]`, and Parquet (`.[data]`/`.[parquet]`). The
router contains 535 built-in routes and reports recognition, extraction and
direct-training capability separately.

201 text/code/structured formats use the built-in bounded-memory text extractor.
Binary document, image, audio, video and 3D routes remain recognition-only until
their reviewed decoders and modality trainers are implemented.

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
path. Image/audio/video/3D routes are recognized and classified, but the base
edition does not yet claim multimodal training.
