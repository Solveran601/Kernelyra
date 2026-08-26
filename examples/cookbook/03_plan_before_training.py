"""Inspect the resolved plan before allocating a training run."""
from kernelyra import AutoTrainer

with AutoTrainer(".kernelyra-plan") as trainer:
    plan = trainer.plan(
        "data/train.csv",
        target="label",
        execution="cpu",
        algorithm_pack="careful",
        cpu=50,
        ram=45,
        threads=2,
        max_steps=2_000,
    )
print(plan.to_dict())
assert plan.ram <= 50
