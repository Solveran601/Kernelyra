// Kernelyra memory kernels. The exported surface is a dependency-free C ABI so
// C, C++, Rust, Go, C# and Python can share the same buffers without copies.

const allocator = @import("memory_allocator.zig");
const guard = @import("memory_guard.zig");
const normalize = @import("memory_normalize.zig");
const transfer = @import("memory_transfer.zig");

export fn kr_zig_alloc_aligned(bytes: usize, alignment: usize) ?*anyopaque {
    return allocator.alloc(bytes, alignment);
}

export fn kr_zig_free_aligned(pointer: ?*anyopaque) void {
    allocator.free(pointer);
}

export fn kr_zig_normalize_f32(
    data: [*]f32,
    rows: usize,
    features: usize,
    means: [*]const f32,
    stds: [*]const f32,
) void { normalize.normalizeF32(data, rows, features, means, stds); }

export fn kr_zig_copy_f32(destination: [*]f32, source: [*]const f32, values: usize) void { transfer.copyF32(destination, source, values); }

export fn kr_zig_zero_f32(destination: [*]f32, values: usize) void { transfer.zeroF32(destination, values); }

/// Return 1 only when every value is finite.  This is used after every native
/// update so a NaN/Inf never reaches an exported checkpoint.
export fn kr_zig_all_finite_f32(values: [*]const f32, count: usize) u32 { return guard.allFiniteF32(values, count); }

/// Clamp a buffer in place for explicit recovery tools and diagnostics.
export fn kr_zig_clip_f32(values: [*]f32, count: usize, limit: f32) void { guard.clipF32(values, count, limit); }
