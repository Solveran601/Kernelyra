from kernelyra import RunConfig, Workspace

workspace = Workspace.open("./example-workspace")
run = workspace.create_run(RunConfig(
    name="customers-v1",
    dataset="demo",
    backend="numpy",
    target_metric=.85,
    batch_mode="auto",
))
print("draft:", run.info.id, run.info.batch_size)
run.start()
