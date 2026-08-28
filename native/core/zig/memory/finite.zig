//! Non-finite detection kept separate from recovery policy.

const std = @import("std");

pub fn allFiniteF32(values: [*]const f32, count: usize) u32 {
    @setRuntimeSafety(false);
    var index: usize = 0;
    while (index < count) : (index += 1) {
        if (!std.math.isFinite(values[index])) return 0;
    }
    return 1;
}
