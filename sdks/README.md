# Kernelyra SDKs

The primary distribution SDKs are Python, Go, C++17, Rust and C#/.NET 8. They
share one local engine and the same easy configuration vocabulary.

| Language | Package source | Easy call | Build metadata |
|---|---|---|---|
| Python | `src/kernelyra` | `fit(dataset, target)` | `pyproject.toml` |
| Go | `sdks/go` | `client.Fit(dataset, target, config)` | `go.mod` |
| C++17 | `sdks/cpp` | `client.fit(dataset, target, config)` | `CMakeLists.txt` |
| Rust | `sdks/rust` | `client.fit(dataset, target, config)` | `Cargo.toml` |
| C# | `sdks/csharp` | `client.Fit(dataset, target, config)` | `.csproj` |

Additional protocol adapters for C, Java, Kotlin, PHP, Ruby and Swift remain in
the tree, but the five SDKs above receive the strongly typed easy API first.

All non-Python clients run a persistent `kernelyra --workspace PATH rpc` child.
The stable `kernelyra-jsonl/1` protocol accepts `ping`, `capabilities`,
`hardware`, `plan`, `train` and `finetune`. Requests are bounded to 1 MiB.

SDKs never reimplement training, batching or safety logic. This guarantees that
an automatic plan chosen from Go is the same plan chosen from Python or C#.

From the repository root, verify all five APIs with real two-step training:

```console
python scripts/smoke_sdks.py
```
