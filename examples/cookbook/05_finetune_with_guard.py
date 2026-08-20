"""Fine-tune while preserving the pre-finetune checkpoint if validation degrades."""
from kernelyra import Config, Engine

settings = (
    Config().target("label").optimizer(learning_rate=1e-4, weight_decay=1e-5)
    .quality(evaluation_interval=100, early_stopping_patience=10)
    .guard(margin=.015, patience=2).steps(4_000)
)
with Engine(".kernelyra-finetune") as engine:
    result = engine.finetune("models/base.npz", "data/domain.csv", settings=settings)
print(result.run.metrics["health"])
