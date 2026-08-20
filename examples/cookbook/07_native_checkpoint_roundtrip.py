"""Export native parameters, restore them into a second model, verify predictions."""
import numpy as np
from kernelyra.native_core import NativeModel

x = np.asarray([[-2., -1.], [-1., -2.], [1., 2.], [2., 1.]], dtype=np.float32)
y = np.asarray([0, 0, 1, 1], dtype=np.float32)
with NativeModel(task="binary_classification", features=2, seed=42) as trained:
    trained.train_random_steps(x, y, batch_size=4, steps=200)
    weights, bias = trained.export_parameters()
    expected = trained.predict(x)
with NativeModel(task="binary_classification", features=2, seed=1) as restored:
    restored.import_parameters(weights, bias)
    np.testing.assert_allclose(restored.predict(x), expected)
print("checkpoint round-trip is exact")
