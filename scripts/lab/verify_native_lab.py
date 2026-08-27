"""Verify local experimental native components without replacing the package DLL."""

from __future__ import annotations

import json

import numpy as np

from kernelyra.native_core import NativeCore, native_core_status


def main() -> int:
    core = NativeCore()
    values = np.asarray([[1.0, np.nan], [np.inf, -2.0]], dtype=np.float32)
    repaired, repaired_count = core.preprocess_f32(
        values, np.asarray([0.0, 2.0], dtype=np.float32), np.asarray([1.0, 2.0], dtype=np.float32), clip_limit=3.0
    )
    chunks = core.plan_text_chunks("One sentence. Another sentence preserves context.", minimum_bytes=12, target_bytes=24, maximum_bytes=36, overlap_bytes=8)
    zig_values = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    core.fill_f32(zig_values, 2.0)
    core.scale_f32(zig_values, 0.5)
    core.add_f32(zig_values, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    moments = core.moments(np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    softmax = core.softmax(np.asarray([-2.0, 0.0, 1.0], dtype=np.float32))
    clipped, observed_norm = core.clip_l2(np.asarray([3.0, 4.0], dtype=np.float32), 2.0)
    print(json.dumps({
        "native": native_core_status(),
        "repaired_values": repaired.tolist(),
        "repaired_count": repaired_count,
        "zig_memory": {"values": zig_values.tolist(), "sum_max_abs": core.memory_summary(zig_values)},
        "fortran_numeric": {
            "moments": moments,
            "softmax": softmax.tolist(),
            "softmax_sum": float(softmax.sum()),
            "clip_l2_observed": observed_norm,
            "clip_l2_values": clipped.tolist(),
        },
        "text_chunks": chunks,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
