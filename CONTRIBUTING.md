# Contributing

1. Use Python 3.11 or newer.
2. Install with `python -m pip install -e ".[tensorflow,mcp,dev]"`.
3. Run `python scripts/release.py check` before submitting changes.
4. Keep `import kernelyra` free of filesystem, thread, network and TensorFlow side effects.
5. Add new ML engines and file handlers as reviewed built-ins with explicit capability tests.
6. Do not claim a format is trainable unless an integration test covers decoding and training.
7. Keep API, CLI, MCP and OpenAPI changes backward compatible within `/api/v1`.
8. A stable release requires the full 60-minute soak workflow on Windows x64.
