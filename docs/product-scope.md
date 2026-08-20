# Product scope

Kernelyra 0.3 is a terminal-first training engine and embeddable library. Its
stable core covers tabular binary classification, multiclass classification and
regression through PyTorch, TensorFlow/Keras and NumPy.

Core trainable inputs are CSV, TSV, flat JSONL/NDJSON, numeric NPZ and Parquet.
Large text-tabular and Parquet sources use bounded-memory streaming. Image,
audio, video, databases and distributed training have built-in format contracts,
but their base-edition trainers are not implemented; the router does not turn
recognition into a training claim.

There is intentionally no graphical training UI. Supported training entry
surfaces are the CLI, Python API, JSONL stdio protocol, optional headless
HTTP/MCP gateways, and the adapters under `sdks/`. There is no plugin center or
dynamic third-party code loading.
