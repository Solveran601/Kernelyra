"""Inspect native capabilities and call finite/norm/clip safety kernels."""
import numpy as np
from kernelyra.native_core import NativeCore, native_core_status

core = NativeCore()
values = np.asarray([-5.0, 3.0, np.inf], dtype=np.float32)
print(native_core_status())
print("finite:", core.all_finite(values))
safe = core.clip(np.nan_to_num(values, nan=0.0, posinf=4.0), limit=2.0)
print("clipped:", safe, "l2:", core.l2_norm(safe))
