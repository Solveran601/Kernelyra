//! Aligned allocation boundary for the native tensor arena.

const alignment_rules = @import("alignment.zig");

extern fn _aligned_malloc(size: usize, alignment: usize) ?*anyopaque;
extern fn _aligned_free(pointer: ?*anyopaque) void;

pub fn alloc(bytes: usize, alignment: usize) ?*anyopaque {
    if (bytes == 0 or !alignment_rules.isValid(alignment)) return null;
    return _aligned_malloc(bytes, alignment);
}

pub fn free(pointer: ?*anyopaque) void {
    if (pointer != null) _aligned_free(pointer);
}
