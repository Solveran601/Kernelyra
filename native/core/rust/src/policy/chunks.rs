use crate::hash::mix_u64;

/// Pick a reproducible but non-uniform bounded chunk size.
pub fn next(
    remaining_records: usize,
    target_records: usize,
    minimum_records: usize,
    maximum_records: usize,
    sequence: u64,
    seed: u64,
) -> usize {
    if remaining_records == 0
        || target_records == 0
        || minimum_records == 0
        || maximum_records < minimum_records
    {
        return 0;
    }
    let lower = (target_records.saturating_mul(3) / 4).clamp(minimum_records, maximum_records);
    let upper = (target_records.saturating_mul(5) / 4).clamp(lower, maximum_records);
    let proposed = lower + (mix_u64(seed ^ sequence) as usize % (upper - lower + 1));
    proposed.min(remaining_records).max(remaining_records.min(minimum_records))
}
