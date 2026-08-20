from __future__ import annotations

import argparse
import time
from pathlib import Path

from kernelyra import RunConfig, Workspace

parser = argparse.ArgumentParser()
parser.add_argument("--workspace", default="./multiclass-example-workspace")
args = parser.parse_args()
root = Path(args.workspace).resolve()
source = root / "multiclass.jsonl"
root.mkdir(parents=True, exist_ok=True)
source.write_text("\n".join(f'{{"x1":{i},"x2":{i % 5},"target":"class-{i % 3}"}}' for i in range(180)), encoding="utf-8")

with Workspace.open(root) as workspace:
    dataset = workspace.datasets.import_file(source, "target")
    run = workspace.create_run(RunConfig(dataset=dataset.id, backend="numpy", objective="multiclass_classification", target_metric=.5, max_steps=100)).start()
    while run.status not in {"completed", "error_recoverable"}:
        time.sleep(.1)
        run = workspace.runs.get(run.id).info
    print(run.to_dict())
