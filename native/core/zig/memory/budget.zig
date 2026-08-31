//! Exact bounded-memory planning for row-major float32 training batches.
//!
//! A batch has one feature buffer and one target buffer.  Buffers are leased
//! from 64-byte aligned arenas, so the planner accounts for the alignment gap
//! between them as well as the values themselves.  It performs no allocation.

const std = @import("std");

pub const default_alignment: usize = 64;

pub const BatchPlan = extern struct {
    capacity_bytes: usize,
    requested_rows: usize,
    planned_rows: usize,
    maximum_rows: usize,
    features: usize,
    buffer_count: usize,
    bytes_per_row: usize,
    required_bytes: usize,
    fits: u32,
};

fn multiply(left: usize, right: usize) ?usize {
    return std.math.mul(usize, left, right) catch null;
}

fn add(left: usize, right: usize) ?usize {
    return std.math.add(usize, left, right) catch null;
}

pub fn makeBatchPlan(
    capacity_bytes: usize,
    requested_rows: usize,
    features: usize,
    buffer_count: usize,
) ?BatchPlan {
    if (capacity_bytes == 0 or requested_rows == 0 or features == 0 or buffer_count < 2) return null;

    const values_per_row = add(features, 1) orelse return null;
    const bytes_per_row = multiply(values_per_row, @sizeOf(f32)) orelse return null;
    // The first lease starts on the already-aligned arena base.  Every later
    // lease can need up to alignment-1 bytes of padding; reserve alignment to
    // keep this integer-only plan conservative.
    const alignment_reserve = multiply(buffer_count - 1, default_alignment) orelse return null;
    if (capacity_bytes <= alignment_reserve) return null;
    const usable_bytes = capacity_bytes - alignment_reserve;
    const maximum_rows = usable_bytes / bytes_per_row;
    if (maximum_rows == 0) return null;
    const value_bytes = multiply(requested_rows, bytes_per_row) orelse return null;
    const required_bytes = add(value_bytes, alignment_reserve) orelse return null;
    const planned_rows = @min(requested_rows, maximum_rows);
    return .{
        .capacity_bytes = capacity_bytes,
        .requested_rows = requested_rows,
        .planned_rows = planned_rows,
        .maximum_rows = maximum_rows,
        .features = features,
        .buffer_count = buffer_count,
        .bytes_per_row = bytes_per_row,
        .required_bytes = required_bytes,
        .fits = if (required_bytes <= capacity_bytes) 1 else 0,
    };
}
