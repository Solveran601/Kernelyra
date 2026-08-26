//! Simple in-place float32 arithmetic used for buffer preparation and tests.

pub fn fillF32(destination: [*]f32, values: usize, value: f32) void {
    @setRuntimeSafety(false);
    var index: usize = 0;
    while (index < values) : (index += 1) destination[index] = value;
}

pub fn scaleF32(destination: [*]f32, values: usize, scale: f32) void {
    @setRuntimeSafety(false);
    var index: usize = 0;
    while (index < values) : (index += 1) destination[index] *= scale;
}

pub fn addF32(destination: [*]f32, source: [*]const f32, values: usize) void {
    @setRuntimeSafety(false);
    var index: usize = 0;
    while (index < values) : (index += 1) destination[index] += source[index];
}
