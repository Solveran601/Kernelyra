# Backends

`backend="auto"` selects the first installed compatible backend in this order:
Kernelyra native, PyTorch, TensorFlow, NumPy. Explicit selection never falls
back silently.

- Native: bundled C++17 linear binary/multiclass/regression training, AVX2/FMA
  runtime dispatch with portable fallback, OpenMP for large kernels, NPZ
  checkpoints, and numeric CSV/TSV streaming without materializing the file.
- PyTorch: MLP training, AdamW, CUDA memory fraction, gradient clipping, safe
  `weights_only` `.pt/.pth` fine-tuning, float32/float16/bfloat16.
- TensorFlow/Keras: MLP training, AdamW, bounded GPU memory, safe model loading,
  `.keras/.h5/.hdf5` fine-tuning and mixed precision.
- NumPy: compatibility linear baseline, `.npz` checkpoint fine-tuning and
  float32/float64 operation.

All backends implement the same split, train/evaluate/test, checkpoint and close
contract. Backend packages are lazy imports, so importing `kernelyra` does not
load PyTorch or TensorFlow.

Install one or more backends:

```console
python -m pip install ".[torch]"
python -m pip install ".[tensorflow]"
python -m pip install ".[full]"
kernelyra native status
kernelyra native build  # source checkout only; binary wheels already include it
```

Backends are reviewed built-ins; Python package entry points are not loaded.
