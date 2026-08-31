"""User journey: plan and optionally train through the small Python API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernelyra import Config, Engine


def plan_summary(plan: object) -> dict[str, object]:
    return {
        "dataset": plan.dataset,
        "target": plan.target,
        "task": plan.task,
        "backend": plan.backend,
        "architecture": plan.architecture,
        "model_format": plan.model_format,
        "execution": plan.execution,
        "resources": {"cpu": plan.cpu, "ram": plan.ram, "gpu": plan.gpu, "threads": plan.threads},
        "batch_size": plan.batch_size,
        "data_mode": plan.data_mode,
        "records_estimate": plan.records_estimate,
        "features_estimate": plan.features_estimate,
        "sources": plan.sources,
        "warnings": list(plan.warnings),
    }


def result_summary(result: object) -> dict[str, object]:
    return {
        "checkpoint": result.checkpoint,
        "dataset": {
            "id": result.dataset.id,
            "records": result.dataset.records,
            "features": result.dataset.features,
            "warnings": result.dataset.warnings,
        },
        "run": {
            "id": result.run.id,
            "status": result.run.status,
            "backend": result.run.effective_backend,
            "architecture": result.run.architecture,
            "task": result.run.objective,
            "best_score": result.run.best_score,
            "best_step": result.run.best_step,
            "termination_reason": result.run.termination_reason,
            "metrics": result.run.metrics,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("target")
    parser.add_argument("--workspace", type=Path, default=Path(".kernelyra-user-journeys"))
    parser.add_argument("--backend", default="auto", choices=("auto", "native", "numpy", "torch", "tensorflow"))
    parser.add_argument("--execution", default="cpu", choices=("auto", "cpu", "hybrid"))
    parser.add_argument("--cpu", type=int, default=80)
    parser.add_argument("--ram", type=int, default=70)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--train", action="store_true", help="Actually create and run a training job.")
    args = parser.parse_args()

    if not args.dataset.is_file():
        parser.error(f"dataset does not exist: {args.dataset}")

    settings = (
        Config()
        .target(args.target)
        .backend(args.backend)
        .execution(args.execution)
        .resources(cpu=args.cpu, ram=args.ram, gpu=0, threads=args.threads)
        .steps(args.steps)
    )

    with Engine(args.workspace) as kernelyra:
        planned = kernelyra.plan(args.dataset, settings=settings)
        payload: dict[str, object] = {
            "requested_settings": settings.to_dict(),
            "resolved_plan": plan_summary(planned),
            "training_started": args.train,
        }
        if args.train:
            result = kernelyra.fit(args.dataset, settings=settings)
            payload["result"] = result_summary(result)

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print("\nUX check: was the distinction between reusable Config, plan and fit understandable?")


if __name__ == "__main__":
    main()
