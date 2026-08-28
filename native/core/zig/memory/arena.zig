//! A bounded, monotonic native arena for synchronous training batches.
//!
//! The arena performs one aligned allocation up front.  Individual leases are
//! offset-only: there is no per-batch malloc/free traffic, and ``reset`` or
//! ``rewind`` returns capacity in O(1).  It is deliberately single-owner; a
//! caller must not acquire/reset the same handle concurrently from multiple
//! threads.  Existing views become invalid after reset or a rewind below their
//! offset.

const std = @import("std");
const alignment_rules = @import("alignment.zig");
const allocator = @import("allocator.zig");

pub const Arena = struct {
    base: ?*anyopaque,
    capacity: usize,
    used: usize,
    high_water: usize,
    alignment: usize,
    allocations: u64,
    failed_allocations: u64,
    resets: u64,
};

fn alignForward(value: usize, requested_alignment: usize) ?usize {
    const expanded = std.math.add(usize, value, requested_alignment - 1) catch return null;
    return expanded & ~@as(usize, requested_alignment - 1);
}

pub fn create(capacity_bytes: usize, requested_alignment: usize) ?*Arena {
    if (capacity_bytes == 0 or !alignment_rules.isValid(requested_alignment)) return null;
    const header_memory = allocator.alloc(@sizeOf(Arena), @alignOf(Arena)) orelse return null;
    const arena: *Arena = @ptrCast(@alignCast(header_memory));
    const payload = allocator.alloc(capacity_bytes, requested_alignment) orelse {
        allocator.free(header_memory);
        return null;
    };
    arena.* = .{
        .base = payload,
        .capacity = capacity_bytes,
        .used = 0,
        .high_water = 0,
        .alignment = requested_alignment,
        .allocations = 0,
        .failed_allocations = 0,
        .resets = 0,
    };
    return arena;
}

pub fn destroy(optional_arena: ?*Arena) void {
    const arena = optional_arena orelse return;
    if (arena.base) |base| allocator.free(base);
    allocator.free(@ptrCast(arena));
}

pub fn acquire(optional_arena: ?*Arena, bytes: usize, requested_alignment: usize) ?*anyopaque {
    const arena = optional_arena orelse return null;
    if (bytes == 0 or !alignment_rules.isValid(requested_alignment) or requested_alignment > arena.alignment) {
        arena.failed_allocations +%= 1;
        return null;
    }
    const offset = alignForward(arena.used, requested_alignment) orelse {
        arena.failed_allocations +%= 1;
        return null;
    };
    const next = std.math.add(usize, offset, bytes) catch {
        arena.failed_allocations +%= 1;
        return null;
    };
    if (next > arena.capacity) {
        arena.failed_allocations +%= 1;
        return null;
    }
    const base = arena.base orelse {
        arena.failed_allocations +%= 1;
        return null;
    };
    arena.used = next;
    arena.high_water = @max(arena.high_water, next);
    arena.allocations +%= 1;
    return @ptrFromInt(@intFromPtr(base) + offset);
}

pub fn mark(optional_arena: ?*const Arena) usize {
    const arena = optional_arena orelse return 0;
    return arena.used;
}

pub fn rewind(optional_arena: ?*Arena, offset: usize) bool {
    const arena = optional_arena orelse return false;
    if (offset > arena.used) return false;
    arena.used = offset;
    return true;
}

pub fn reset(optional_arena: ?*Arena) void {
    const arena = optional_arena orelse return;
    arena.used = 0;
    arena.resets +%= 1;
}

pub fn capacityBytes(optional_arena: ?*const Arena) usize {
    const arena = optional_arena orelse return 0;
    return arena.capacity;
}

pub fn used(optional_arena: ?*const Arena) usize {
    const arena = optional_arena orelse return 0;
    return arena.used;
}

pub fn highWater(optional_arena: ?*const Arena) usize {
    const arena = optional_arena orelse return 0;
    return arena.high_water;
}

pub fn arenaAlignment(optional_arena: ?*const Arena) usize {
    const arena = optional_arena orelse return 0;
    return arena.alignment;
}

pub fn allocations(optional_arena: ?*const Arena) u64 {
    const arena = optional_arena orelse return 0;
    return arena.allocations;
}

pub fn failedAllocations(optional_arena: ?*const Arena) u64 {
    const arena = optional_arena orelse return 0;
    return arena.failed_allocations;
}

pub fn resets(optional_arena: ?*const Arena) u64 {
    const arena = optional_arena orelse return 0;
    return arena.resets;
}
