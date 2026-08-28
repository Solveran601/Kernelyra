//! One-pass float32 data repair, standardization and clipping.
//!
//! Keeping these three operations together avoids the previous repair scan
//! followed by a second C++ normalization scan. The ABI caller validates
//! pointers and the clipping limit before this kernel is entered.

const std = @import("std");
const shape = @import("memory_shape.zig");

pub fn preprocessF32(
    data: [*]f32,
    rows: usize,
    features: usize,
    means: [*]const f32,
    stds: [*]const f32,
    clip_limit: f32,
) u64 {
    @setRuntimeSafety(false);
    const total = shape.elements(rows, features) orelse return 0;
    if (total == 0 or features == 0) return 0;

    var repaired: u64 = 0;
    var index: usize = 0;
    while (index < total) : (index += 1) {
        const feature = index % features;
        const raw_mean = means[feature];
        const mean = if (std.math.isFinite(raw_mean)) raw_mean else 0.0;
        const deviation = stds[feature];
        const scale = if (std.math.isFinite(deviation) and
            (deviation > 1.0e-12 or deviation < -1.0e-12)) deviation else 1.0;

        var value = data[index];
        if (!std.math.isFinite(value)) {
            value = mean;
            repaired += 1;
        }
        value = (value - mean) / scale;
        data[index] = @max(-clip_limit, @min(clip_limit, value));
    }
    return repaired;
}
