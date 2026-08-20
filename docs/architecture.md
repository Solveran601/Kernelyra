# Architecture

Kernelyra has one canonical engine and several entry surfaces. The training
engine has no web frontend or JavaScript/TypeScript runtime. Backends, ingestors
and format routes are closed, reviewed built-ins.

```text
CLI / Python API / C,C++,C#,Rust,Go,PHP,Java,Kotlin,Swift,Ruby
                         |
                  JSONL stdio contract
                         |
        AutoTrainer -> dataset router -> hardware/batch planner
                         |
             isolated backend worker process
           /          /        |           \
 native ABI core  PyTorch  TensorFlow      NumPy
 C++ ABI / Fortran training / Zig memory
                         |
       streaming reader + checkpoints + result-driven stop
```

Python owns orchestration because PyTorch, TensorFlow, Pandas and PyArrow expose
their supported APIs there. The native data/training path uses a C++ ABI and
streaming shell, Zig for explicit aligned allocation/copy/zero/normalization,
and Fortran for dense dot, AXPY, gradient and weight-update arithmetic. Rust
can call the same stable C ABI directly. The released native components are
enabled by default; callers can temporarily mask them only to verify fallback
correctness. C, C++ and Rust use the ABI for partial training and memory work.
Duplicating training state machines in every language would create incompatible
results and unsafe resume behavior, so other languages use `kernelyra-jsonl/1`.

Large tabular files and compatible multi-file folders are registered as external immutable sources. A bounded scan
builds schema/statistics, training reads batches on demand, validation/test are
capped, and the checkpoint stores the delivered stream cursor. Source
fingerprints prevent silently resuming on changed data.

The optional HTTP daemon and MCP server are headless automation gateways. They
are not needed by `plan`, `train`, `finetune`, Python `AutoTrainer`, or JSONL RPC.
