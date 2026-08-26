# Native laboratory scripts

These scripts are local development tools; they are not release commands and
do not make public performance or LLM-training claims.

- `build_native_lab.ps1 -Test` builds Rust, Zig, Fortran, C++ and C, then runs
  the native contract test.
- `run_live_showcase.ps1` runs a real small v5 native tabular training job and
  prints CSV/JSONL routing plus experimental text-boundary evidence.
- `native_status.py` shows the exact DLL and its enabled components.
- `plan_training.py DATASET --target LABEL` produces a resource and execution
  plan without starting training.
- `build_native_source.py --output .test_workspaces/manual-native` exercises
  the non-CMake source build without overwriting the packaged DLL.
- `run_native_contract_tests.ps1` configures, builds and executes the ABI
  contract test through CMake.
- `inspect_tabular_ingestion.py DATASET --stream-spec` prints the actual
  ingestion route and bounded streaming contract.
- `inspect_text_chunks.py --file TEXT.txt` prints the Rust UTF-8 chunk ranges.
- `verify_native_lab.py` checks the experimental DLL without overwriting the
  packaged DLL.

The text planner preserves byte-safe context prefixes and contiguous content
ranges. It is **preprocessing only**: a language-model trainer and loss masking
are intentionally not claimed by Kernelyra v5.
