# Library API

Kernelyra uses the same option names and defaults in Python, Go, C++, Rust and
C#. Omitting an option means automatic selection. Explicit values are validated
and used exactly; unsafe batches require an explicit risk acknowledgement.

## Python

```python
from kernelyra import fit

result = fit("train.csv", "label", workspace="./workspace")
print(result.run.status, result.checkpoint)
```

```python
from kernelyra import Config, Engine

config = (
    Config()
    .backend("native")
    .profile("balanced")
    .goal(.95)
    .steps(10_000)
    .batch(64, accept_risk=True)
    .resources(cpu=70, ram=60, gpu=75)
    .optimizer(learning_rate=.0003, weight_decay=.01)
    .model(256, 128, 64, precision="auto")
    .data(workers=4, prefetch=2)
    .quality(evaluation_interval=100, min_improvement=.0005,
             early_stopping_patience=18, target_patience=3)
    .guard(margin=.03, patience=3)
    .seed(42)
)

with Engine("./workspace") as engine:
    plan = engine.plan("train.csv", "label", settings=config)
    result = engine.fit("train.csv", "label", settings=config)
    tuned = engine.finetune("model.pth", "new.csv", "label", settings=config)
```

`Config.set(custom_name=value)` exposes advanced stable protocol options. The
original `train`, `finetune`, `AutoTrainer` and `Workspace` APIs remain intact.
`Config.copy()`, `merge()`, `unset()` and `automatic()` support reusable policy
composition. `Engine` accepts defaults at construction, can update them through
`configure()`, inspects data without importing it, and provides sequential
`plan_many()`/`fit_many()` helpers for several files or folders.

## Go

```go
engine, err := kernelyra.Open("./workspace")
if err != nil { panic(err) }
defer engine.Close()

result, err := engine.Fit("train.csv", "label", nil) // full auto
```

```go
config := kernelyra.Auto("label").
    WithBackend("torch").WithGoal(.95).WithSteps(10_000).
    WithBatch(64, true).WithResources(70, 60, 75).
    WithOptimizer(.0003, .01).WithModel("auto", 256, 128, 64)
result, err := engine.Fit("train.csv", "label", config)
```

Module: `sdks/go`, import path `github.com/kernelyra-ai/kernelyra-go`.

## C++17

```cpp
#include "kernelyra_process.hpp"

auto engine = kernelyra::open("./workspace");
auto result = engine.fit("train.csv", "label");
```

```cpp
auto config = kernelyra::Config::automatic()
    .backend("torch").goal(.95).steps(10000)
    .batch(64, true).resources(70, 60, 75)
    .optimizer(.0003, .01).model({256, 128, 64});
auto result = engine.fit("train.csv", "label", config);
```

The 0.3 header-only process transport is released and tested on Windows x64. CMake target:
`kernelyra_client` under `sdks/cpp`.

## Rust

```rust
let mut engine = Client::open("./workspace")?;
let result = engine.fit("train.csv", "label", None)?;
```

```rust
let config = Config::default().backend("torch").goal(.95).steps(10_000)
    .batch(64, true).resources(70, 60, 75)
    .optimizer(.0003, .01).model(&[256, 128, 64], "auto");
let result = engine.fit("train.csv", "label", Some(config))?;
```

Crate source and an executable example are under `sdks/rust`.

## C# / .NET 8

```csharp
using var engine = new Kernelyra.Client("./workspace");
var result = engine.Fit("train.csv", "label");
```

```csharp
var config = Config.Auto().Backend("torch").Goal(.95).Steps(10_000)
    .Batch(64, acceptRisk: true).Resources(70, 60, 75)
    .Optimizer(.0003, .01).Model("auto", 256, 128, 64);
var result = engine.Fit("train.csv", "label", config);
```

The packable project is `sdks/csharp/Kernelyra.Client.csproj`.

## Shared configuration

The common settings are target/task, backend, hardware profile, goal metric,
maximum steps, batch and risk approval, CPU/RAM/GPU limits, learning rate,
weight decay, hidden layers, precision, data workers, prefetch and seed.

Non-Python SDKs start one persistent local `kernelyra rpc` process. Install the
platform wheel and ensure `kernelyra` is on `PATH`; it already contains the C++
runtime and does not require a compiler. SDKs also expose a custom executable
path for embedded distributions. Training stays in the engine rather than being
reimplemented differently in every language. The native hot-path C ABI is kept
stable separately in `native/include/kernelyra_core.h` for direct embedding when a
host application explicitly needs zero JSONL overhead.
