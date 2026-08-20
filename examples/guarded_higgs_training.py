"""Production-style, resource-bounded HIGGS training with Kernelyra.

Download a flat HIGGS CSV first, then run:
    python examples/guarded_higgs_training.py path\\to\\higgs.csv

The example deliberately sets a ceiling as well as a quality target.  A run can
finish because it reached the target or because the safety ceiling was reached;
the resulting report makes the distinction explicit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernelyra import Config, Engine


def report(run) -> dict[str, object]:
    return {
        "status": run.run.status,
        "message": run.run.message,
        "checkpoint": run.checkpoint,
        "records": run.dataset.records,
        "features": run.dataset.features,
        "backend": run.plan.backend,
        "limits": {"cpu_percent": run.plan.cpu, "ram_percent": run.plan.ram, "gpu": run.plan.gpu},
        "metrics": run.run.metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Flat HIGGS CSV")
    parser.add_argument("--label", default="is_boson", help="Binary target column in the downloaded HIGGS CSV")
    parser.add_argument("--workspace", type=Path, default=Path(".benchmarks/higgs-workspace"))
    parser.add_argument("--ram-percent", type=int, default=50, help="Hard RAM budget as a percent (10..95)")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--target", type=float, default=0.72, help="Validation target metric")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()

    settings = (
        Config()
        .target(args.label)
        .task("binary_classification")
        .hardware("custom", cpu=80, ram=args.ram_percent, gpu=0)
        .data(workers=2, prefetch=3)
        .model(128, 64, precision="float32")
        .optimizer(learning_rate=args.learning_rate, weight_decay=1e-5)
        .stopping(maximum_steps=args.steps, target_metric=args.target, early_stopping_patience=12)
        .quality(evaluation_interval=100, min_improvement=2e-4, target_patience=2)
        .guard(margin=0.015, patience=2)
        .seed(20260820)
    )

    with Engine(args.workspace) as engine:
        result = engine.fit(args.dataset, settings=settings)

    print(json.dumps(report(result), indent=2, sort_keys=True, default=str))
    return 0 if result.run.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
