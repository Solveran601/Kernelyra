# Product scope

Kernelyra 0.3 is a terminal-first training engine and embeddable library. Its
stable core covers tabular binary classification, multiclass classification and
regression through PyTorch, TensorFlow/Keras and NumPy.

Core trainable inputs are CSV, TSV, flat JSONL/NDJSON, numeric NPZ and Parquet.
Large compatible tabular sources use bounded-memory streaming. Kernelyra 0.3
does not include built-in multimodal, database, or distributed trainers.

Supported training entry surfaces are the CLI, Python API, JSONL stdio protocol,
optional headless HTTP/MCP gateways, and the adapters under `sdks/`.
