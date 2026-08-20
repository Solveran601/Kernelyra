from __future__ import annotations

import argparse
import time
from pathlib import Path

from kernelyra import RunConfig, Workspace

parser = argparse.ArgumentParser()
parser.add_argument("--workspace", default="./binary-example-workspace")
args = parser.parse_args()
root = Path(args.workspace).resolve()
source = root / "binary.csv"
root.mkdir(parents=True, exist_ok=True)
source.write_text("x1,x2,target\n" + "\n".join(f"{i},{i % 7},{int(i % 7 > 3)}" for i in range(160)), encoding="utf-8")

with Workspace.open(root) as workspace:
    dataset = workspace.datasets.import_file(source, "target")
    run = workspace.create_run(RunConfig(dataset=dataset.id, backend="numpy", target_metric=.5, max_steps=100)).start()
    while run.status not in {"completed", "error_recoverable"}:
        time.sleep(.1)
        run = workspace.runs.get(run.id).info
    print(run.to_dict())
