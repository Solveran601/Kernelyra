"""Run many deterministic random mini-batches through one native ABI call."""
import numpy as np
from kernelyra.native_core import NativeModel

rng = np.random.default_rng(7)
x = rng.normal(size=(4096, 28)).astype(np.float32)
y = (x[:, 0] - x[:, 1] > 0).astype(np.float32)
with NativeModel(task="binary_classification", features=28, seed=7, learning_rate=.03) as model:
    final_loss = model.train_random_steps(x, y, batch_size=64, steps=5_000)
    accuracy = float(((model.predict(x) >= .5) == y).mean())
print({"loss": final_loss, "accuracy": accuracy})
