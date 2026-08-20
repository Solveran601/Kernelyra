"""Train a large CSV under a low-memory profile; mode is selected automatically."""
from kernelyra import Config, Engine

settings = Config().target("label").low_memory().data(workers=2, prefetch=1).steps(20_000)
with Engine(".kernelyra-stream") as engine:
    result = engine.fit("data/large_train.csv", settings=settings)
print({"data_mode": result.plan.data_mode, "checkpoint": result.checkpoint})
