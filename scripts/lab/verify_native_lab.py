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
    print(json.dumps({"native": native_core_status(), "repaired_values": repaired.tolist(), "repaired_count": repaired_count, "text_chunks": chunks}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
