//! Explicit bounded clipping; invalid limits are rejected rather than guessed.

const std = @import("std");

pub fn clipF32(values: [*]f32, count: usize, limit: f32) void {
    @setRuntimeSafety(false);
    if (!std.math.isFinite(limit) or limit <= 0.0) return;
    var index: usize = 0;
    while (index < count) : (index += 1) {
        values[index] = @max(-limit, @min(limit, values[index]));
    }
}
