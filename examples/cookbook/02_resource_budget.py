"""Set hard CPU/RAM limits while keeping batch size automatic."""
from kernelyra import Config, Engine

settings = Config().target("label").hardware("custom", cpu=60, ram=35, gpu=0).steps(5_000)
with Engine(".kernelyra-budget") as engine:
    result = engine.fit("data/train.csv", settings=settings)
print(result.plan.to_dict())
