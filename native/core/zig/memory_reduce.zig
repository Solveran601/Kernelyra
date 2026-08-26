//! Bounded reductions for diagnostics without allocating a temporary buffer.

pub fn sumF32(values: [*]const f32, count: usize) f32 {
    @setRuntimeSafety(false);
    var total: f64 = 0.0;
    var index: usize = 0;
    while (index < count) : (index += 1) total += @as(f64, values[index]);
    return @floatCast(total);
}

pub fn maxAbsF32(values: [*]const f32, count: usize) f32 {
    @setRuntimeSafety(false);
    var maximum: f32 = 0.0;
    var index: usize = 0;
    while (index < count) : (index += 1) maximum = @max(maximum, @abs(values[index]));
    return maximum;
}
