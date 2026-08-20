"""Automatic binary classification from a CSV with a `label` column."""
from kernelyra import fit

result = fit("data/train.csv", "label", workspace=".kernelyra-demo")
print(result.checkpoint)
print(result.run.metrics["test"])
