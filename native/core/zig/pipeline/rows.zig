//! Gather selected row-major feature vectors into an already-allocated batch.

const shape = @import("../memory/shape.zig");

pub fn gatherRowsF32(
    source: [*]const f32,
    source_rows: usize,
    features: usize,
    selected: [*]const usize,
    selected_rows: usize,
    destination: [*]f32,
) bool {
    @setRuntimeSafety(false);
    _ = shape.elements(source_rows, features) orelse return false;
    _ = shape.elements(selected_rows, features) orelse return false;
    var output_row: usize = 0;
    while (output_row < selected_rows) : (output_row += 1) {
        const input_row = selected[output_row];
        if (input_row >= source_rows) return false;
        const input_offset = input_row * features;
        const output_offset = output_row * features;
        var feature: usize = 0;
        while (feature < features) : (feature += 1) destination[output_offset + feature] = source[input_offset + feature];
    }
    return true;
}
