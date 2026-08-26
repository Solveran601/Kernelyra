//! Alignment validation for arena allocations shared across the C ABI.

pub fn isValid(alignment: usize) bool {
    return alignment >= @sizeOf(usize) and (alignment & (alignment - 1)) == 0;
}
