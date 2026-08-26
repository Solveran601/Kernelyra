# Kernelyra SDKs

The native SDK foundation is C, C++17 and Rust. They use the same local engine,
the same JSONL protocol and explicit execution vocabulary: `cpu` or `hybrid`,
an algorithm pack, plus CPU/RAM/GPU/thread limits set by the developer.

| Language | Package source | Main call | Build metadata |
|---|---|---|---|
| C | `sdks/c` | `kernelyra_train_with_options(...)` | header-only |
| C++17 | `sdks/cpp` | `client.fit(dataset, target, config)` | `CMakeLists.txt` |
| Rust | `sdks/rust` | `client.fit(dataset, target, config)` | `Cargo.toml` |

The Python package remains the reference CLI and local engine. Other adapters
that may exist in the source tree are not part of this native SDK baseline.
PHP and Kotlin clients are intentionally not distributed.

All non-Python clients run a persistent `kernelyra --workspace PATH rpc` child.
The stable `kernelyra-jsonl/1` protocol accepts `ping`, `capabilities`,
`hardware`, `plan`, `train` and `finetune`. Requests are bounded to 1 MiB.

SDKs never reimplement training, batching or safety logic. A plan produced
through C, C++ or Rust is therefore evaluated by the same core as a Python CLI
request.

```cpp
auto config = kernelyra::Config::automatic()
    .cpu_only()
    .algorithm_pack(kernelyra::AlgorithmPack::throughput)
    .resources(90, 80, 0, 8);
```

```rust
let config = Config::default()
    .cpu_only()
    .algorithm_pack(AlgorithmPack::Throughput)
    .resources(90, 80, 0)
    .threads(8);
```

```c
kernelyra_run_options options = {
    .target = "label", .execution = KERNELYRA_EXECUTION_CPU,
    .algorithm_pack = KERNELYRA_PACK_THROUGHPUT,
    .cpu = 90, .ram = 80, .threads = 8
};
kernelyra_train_with_options(&client, "train.csv", &options, response, sizeof(response));
```

From the repository root, verify the maintained cross-language smoke suite.
The C check validates JSONL request serialization; C++ and Rust run real
two-step local training through the engine:

```console
python scripts/smoke_sdks.py
```
