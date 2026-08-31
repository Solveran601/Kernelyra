//! Checked shape arithmetic.  Native entry points receive dimensions from
//! foreign callers, so multiplication must not wrap before a buffer walk.

const std = @import("std");

pub fn elements(rows: usize, features: usize) ?usize {
    return std.math.mul(usize, rows, features) catch null;
}
