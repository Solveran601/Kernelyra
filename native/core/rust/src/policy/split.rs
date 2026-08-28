use crate::hash::mix_u64;

pub const TRAIN: u32 = 0;
pub const VALIDATION: u32 = 1;
pub const TEST: u32 = 2;
pub const INVALID: u32 = u32::MAX;

/// Keep every record with the same context key in a single evaluation split.
pub fn for_context(key: u64, validation_percent: u32, test_percent: u32) -> u32 {
    if validation_percent > 95 || test_percent > 95 || validation_percent + test_percent > 95 {
        return INVALID;
    }
    let bucket = (mix_u64(key) % 100) as u32;
    if bucket < validation_percent {
        VALIDATION
    } else if bucket < validation_percent + test_percent {
        TEST
    } else {
        TRAIN
    }
}
