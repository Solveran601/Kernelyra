// Kernelyra memory kernels. The exported surface is a dependency-free C ABI so
// C, C++, Rust, Go, C# and Python can share the same buffers without copies.

const allocator = @import("memory/allocator.zig");
const arena = @import("memory/arena.zig");
const budget = @import("memory/budget.zig");
const arithmetic = @import("memory/arithmetic.zig");
const batch = @import("pipeline/batch.zig");
const guard = @import("pipeline/guard.zig");
const normalize = @import("pipeline/normalize.zig");
const preprocess = @import("pipeline/preprocess.zig");
const reduce = @import("pipeline/reduce.zig");
const repair = @import("pipeline/repair.zig");
const row_gather = @import("pipeline/rows.zig");
const transfer = @import("memory/transfer.zig");

export fn kr_zig_alloc_aligned(bytes: usize, alignment: usize) ?*anyopaque {
    return allocator.alloc(bytes, alignment);
}

export fn kr_zig_free_aligned(pointer: ?*anyopaque) void {
    allocator.free(pointer);
}

export fn kr_zig_arena_create(capacity: usize, alignment: usize) ?*arena.Arena {
    return arena.create(capacity, alignment);
}

export fn kr_zig_arena_destroy(handle: ?*arena.Arena) void {
    arena.destroy(handle);
}

export fn kr_zig_arena_acquire(handle: ?*arena.Arena, bytes: usize, alignment: usize) ?*anyopaque {
    return arena.acquire(handle, bytes, alignment);
}

export fn kr_zig_arena_mark(handle: ?*const arena.Arena) usize {
    return arena.mark(handle);
}

export fn kr_zig_arena_rewind(handle: ?*arena.Arena, offset: usize) u32 {
    return if (arena.rewind(handle, offset)) 1 else 0;
}

export fn kr_zig_arena_reset(handle: ?*arena.Arena) void {
    arena.reset(handle);
}

export fn kr_zig_arena_capacity(handle: ?*const arena.Arena) usize {
    return arena.capacityBytes(handle);
}
export fn kr_zig_arena_used(handle: ?*const arena.Arena) usize {
    return arena.used(handle);
}
export fn kr_zig_arena_high_water(handle: ?*const arena.Arena) usize {
    return arena.highWater(handle);
}
export fn kr_zig_arena_alignment(handle: ?*const arena.Arena) usize {
    return arena.arenaAlignment(handle);
}
export fn kr_zig_arena_allocations(handle: ?*const arena.Arena) u64 {
    return arena.allocations(handle);
}
export fn kr_zig_arena_failed_allocations(handle: ?*const arena.Arena) u64 {
    return arena.failedAllocations(handle);
}
export fn kr_zig_arena_resets(handle: ?*const arena.Arena) u64 {
    return arena.resets(handle);
}

export fn kr_zig_batch_plan_make(
    capacity_bytes: usize,
    requested_rows: usize,
    features: usize,
    buffer_count: usize,
    output: ?*budget.BatchPlan,
) u32 {
    const plan = budget.makeBatchPlan(capacity_bytes, requested_rows, features, buffer_count) orelse return 0;
    const destination = output orelse return 0;
    destination.* = plan;
    return 1;
}

export fn kr_zig_normalize_f32(
    data: [*]f32,
    rows: usize,
    features: usize,
    means: [*]const f32,
    stds: [*]const f32,
) void {
    normalize.normalizeF32(data, rows, features, means, stds);
}

/// Fused matrix preprocessing used when repair, normalization and clipping are
/// all requested. It traverses the source matrix exactly once.
export fn kr_zig_preprocess_f32(
    data: [*]f32,
    rows: usize,
    features: usize,
    means: [*]const f32,
    stds: [*]const f32,
    clip_limit: f32,
) u64 {
    return preprocess.preprocessF32(data, rows, features, means, stds, clip_limit);
}

export fn kr_zig_copy_f32(destination: [*]f32, source: [*]const f32, values: usize) void {
    transfer.copyF32(destination, source, values);
}

export fn kr_zig_zero_f32(destination: [*]f32, values: usize) void {
    transfer.zeroF32(destination, values);
}

export fn kr_zig_fill_f32(destination: [*]f32, values: usize, value: f32) void {
    arithmetic.fillF32(destination, values, value);
}

export fn kr_zig_scale_f32(destination: [*]f32, values: usize, scale: f32) void {
    arithmetic.scaleF32(destination, values, scale);
}

export fn kr_zig_add_f32(destination: [*]f32, source: [*]const f32, values: usize) void {
    arithmetic.addF32(destination, source, values);
}

export fn kr_zig_sum_f32(values: [*]const f32, count: usize) f32 {
    return reduce.sumF32(values, count);
}

export fn kr_zig_max_abs_f32(values: [*]const f32, count: usize) f32 {
    return reduce.maxAbsF32(values, count);
}

export fn kr_zig_repair_nonfinite_f32(
    data: [*]f32,
    rows_count: usize,
    features: usize,
    means: [*]const f32,
) u64 {
    return repair.repairNonFiniteF32(data, rows_count, features, means);
}

export fn kr_zig_gather_rows_f32(
    source: [*]const f32,
    source_rows: usize,
    features: usize,
    selected: [*]const usize,
    selected_rows: usize,
    destination: [*]f32,
) u32 {
    return if (row_gather.gatherRowsF32(source, source_rows, features, selected, selected_rows, destination)) 1 else 0;
}

/// Gather feature rows and their matching targets in one bounded traversal.
/// This keeps labels synchronized with Rust-selected row indices without a
/// second C++ loop over the sampled minibatch.
export fn kr_zig_gather_batch_f32(
    source: [*]const f32,
    targets: [*]const f32,
    source_rows: usize,
    features: usize,
    selected: [*]const usize,
    selected_rows: usize,
    destination: [*]f32,
    destination_targets: [*]f32,
) u32 {
    return if (batch.gatherBatchF32(
        source,
        targets,
        source_rows,
        features,
        selected,
        selected_rows,
        destination,
        destination_targets,
    )) 1 else 0;
}

/// Return 1 only when every value is finite.  This is used after every native
/// update so a NaN/Inf never reaches an exported checkpoint.
export fn kr_zig_all_finite_f32(values: [*]const f32, count: usize) u32 {
    return guard.allFiniteF32(values, count);
}

/// Clamp a buffer in place for explicit recovery tools and diagnostics.
export fn kr_zig_clip_f32(values: [*]f32, count: usize, limit: f32) void {
    guard.clipF32(values, count, limit);
}
