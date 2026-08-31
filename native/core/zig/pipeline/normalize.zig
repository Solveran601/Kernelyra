//! Cache-friendly row-major float32 standardization.

const shape = @import("../memory/shape.zig");

pub fn normalizeF32(
    data: [*]f32,
    rows: usize,
    features: usize,
    means: [*]const f32,
    stds: [*]const f32,
) void {
    @setRuntimeSafety(false);
    const total = shape.elements(rows, features) orelse return;
    if (total == 0) return;
    var row: usize = 0;
    while (row < rows) : (row += 1) {
        const offset = row * features;
        var feature: usize = 0;
        while (feature < features) : (feature += 1) {
            data[offset + feature] = (data[offset + feature] - means[feature]) / stds[feature];
        }
    }
}
