//! Non-finite replacement before normalization.  Means are indexed by column
//! so the operation has the same row-major contract as the C++ preprocessor.

const std = @import("std");
const shape = @import("memory_shape.zig");

pub fn repairNonFiniteF32(data: [*]f32, rows: usize, features: usize, means: [*]const f32) u64 {
    @setRuntimeSafety(false);
    const total = shape.elements(rows, features) orelse return 0;
    if (total == 0 or features == 0) return 0;
    var repaired: u64 = 0;
    var index: usize = 0;
    while (index < total) : (index += 1) {
        if (!std.math.isFinite(data[index])) {
            const mean = means[index % features];
            data[index] = if (std.math.isFinite(mean)) mean else 0.0;
            repaired += 1;
        }
    }
    return repaired;
}
