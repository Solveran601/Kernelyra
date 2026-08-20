// Kernelyra memory kernels. The exported surface is a dependency-free C ABI so
// C, C++, Rust, Go, C# and Python can share the same buffers without copies.

extern fn _aligned_malloc(size: usize, alignment: usize) ?*anyopaque;
extern fn _aligned_free(pointer: ?*anyopaque) void;

export fn kr_zig_alloc_aligned(bytes: usize, alignment: usize) ?*anyopaque {
    if (bytes == 0 or alignment < @sizeOf(usize) or (alignment & (alignment - 1)) != 0) return null;
    return _aligned_malloc(bytes, alignment);
}

export fn kr_zig_free_aligned(pointer: ?*anyopaque) void {
    if (pointer != null) _aligned_free(pointer);
}

export fn kr_zig_normalize_f32(
    data: [*]f32,
    rows: usize,
    features: usize,
    means: [*]const f32,
    stds: [*]const f32,
) void {
    @setRuntimeSafety(false);
    var row: usize = 0;
    while (row < rows) : (row += 1) {
        const offset = row * features;
        var feature: usize = 0;
        while (feature < features) : (feature += 1) {
            data[offset + feature] = (data[offset + feature] - means[feature]) / stds[feature];
        }
    }
}

export fn kr_zig_copy_f32(destination: [*]f32, source: [*]const f32, values: usize) void {
    @setRuntimeSafety(false);
    var index: usize = 0;
    while (index < values) : (index += 1) destination[index] = source[index];
}

export fn kr_zig_zero_f32(destination: [*]f32, values: usize) void {
    @setRuntimeSafety(false);
    var index: usize = 0;
    while (index < values) : (index += 1) destination[index] = 0.0;
}
