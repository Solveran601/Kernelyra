from kernelyra import Config, Engine, fit

# Simplest API: everything except dataset and target is automatic.
automatic = fit("train.csv", "label", workspace="./workspace")
print(automatic.checkpoint)

# Full control remains readable and reusable.
settings = (
    Config()
    .backend("torch")
    .goal(.95)
    .steps(10_000)
    .batch(64, accept_risk=True)
    .resources(cpu=70, ram=60, gpu=75)
    .optimizer(learning_rate=.0003, weight_decay=.01)
    .model(256, 128, 64, precision="auto")
)

with Engine("./workspace") as engine:
    result = engine.fit("train.csv", "label", settings=settings)
    print(result.run.metrics, result.checkpoint)
