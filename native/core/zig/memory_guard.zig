//! Facade joining the finite-value detector and explicit clipping primitive.

const clip = @import("memory_clip.zig");
const finite = @import("memory_finite.zig");

pub const allFiniteF32 = finite.allFiniteF32;

pub const clipF32 = clip.clipF32;
