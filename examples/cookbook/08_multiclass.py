"""Explicit multiclass training with a controlled resource budget."""
from kernelyra import Config, Engine

settings = (
    Config().target("species").task("multiclass_classification")
    .hardware("custom", cpu=70, ram=45, gpu=0).model(128, 64, precision="float32")
    .stopping(maximum_steps=6_000, target_metric=.85, target_patience=2)
)
with Engine(".kernelyra-multiclass") as engine:
    result = engine.fit("data/species.csv", settings=settings)
print(result.run.metrics["test"])
