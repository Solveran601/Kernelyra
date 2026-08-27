//! Explicit float32 copy and zero loops used by C++ ABI fallbacks.

pub fn copyF32(destination: [*]f32, source: [*]const f32, values: usize) void {
    @setRuntimeSafety(false);
    var index: usize = 0;
    while (index < values) : (index += 1) destination[index] = source[index];
}

pub fn zeroF32(destination: [*]f32, values: usize) void {
    @setRuntimeSafety(false);
    var index: usize = 0;
    while (index < values) : (index += 1) destination[index] = 0.0;
}
