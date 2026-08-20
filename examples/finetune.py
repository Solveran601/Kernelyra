from kernelyra import AutoTrainer

with AutoTrainer("./workspace") as engine:
    result = engine.finetune(
        "./model.pth",
        "./train.csv",
        backend="torch",
        target="label",
        learning_rate=0.0001,
        batch_size=32,
    )
    print(result.to_dict())
