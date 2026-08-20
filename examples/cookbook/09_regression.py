"""Regression with result-driven stopping rather than a fixed epoch count."""
from kernelyra import Config, Engine

settings = (
    Config().target("price").task("regression").balanced()
    .optimizer(learning_rate=.005, weight_decay=1e-5)
    .stopping(maximum_steps=10_000, target_metric=.15, early_stopping_patience=15)
    .quality(evaluation_interval=100, min_improvement=1e-4)
)
with Engine(".kernelyra-regression") as engine:
    result = engine.fit("data/housing.csv", settings=settings)
print(result.run.metrics["test"])
