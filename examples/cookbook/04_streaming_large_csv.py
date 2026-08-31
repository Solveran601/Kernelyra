"""Train a large CSV with explicit CPU limits and bounded streaming."""
from kernelyra import Config, Engine

settings = (
    Config()
    .target("label")
    .cpu_only()
    .resources(cpu=50, ram=45, threads=2)
    .data(workers=2, prefetch=1)
    .steps(20_000)
)
with Engine(".kernelyra-stream") as engine:
    result = engine.fit("data/large_train.csv", settings=settings)
print({"data_mode": result.plan.data_mode, "checkpoint": result.checkpoint})
