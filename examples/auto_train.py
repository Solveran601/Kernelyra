from kernelyra import AutoTrainer


def report(run: object) -> None:
    print(
        f"\rstatus={run.status} step={run.step}/{run.max_steps} "
        f"score={run.best_score:.4f} batch={run.batch_size}",
        end="",
        flush=True,
    )


with AutoTrainer("./workspace") as engine:
    resolved = engine.plan("./train.csv", target="label")
    print(resolved.to_dict())
    result = engine.train("./train.csv", target="label", progress=report)
    print("\ncheckpoint:", result.checkpoint)
