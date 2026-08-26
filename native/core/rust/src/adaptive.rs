//! Resource-aware extension of the deterministic chunk policy.
//!
//! The caller supplies two explicit signals instead of a guessed hardware
//! profile: memory pressure (0 = unconstrained, 100 = strict) and aggression
//! (0 = stable/careful, 100 = throughput-first).  The result is still bounded,
//! deterministic and never allocates.

use crate::chunks;

pub fn next(
    remaining_records: usize,
    target_records: usize,
    minimum_records: usize,
    maximum_records: usize,
    sequence: u64,
    seed: u64,
    memory_pressure_percent: u32,
    aggression_percent: u32,
) -> usize {
    if remaining_records == 0
        || target_records == 0
        || minimum_records == 0
        || maximum_records < minimum_records
    {
        return 0;
    }

    let pressure = memory_pressure_percent.min(100) as usize;
    let aggression = aggression_percent.min(100) as usize;
    // Full memory pressure contracts the target to 45%. High aggression then
    // restores at most 35 percentage points; bounds remain authoritative.
    let pressure_scale = 45 + ((100 - pressure) * 55) / 100;
    let aggression_scale = 100 + (aggression * 35) / 100;
    let scaled_target = target_records
        .saturating_mul(pressure_scale)
        .saturating_mul(aggression_scale)
        / 10_000;
    let effective_target = scaled_target.clamp(minimum_records, maximum_records);
    chunks::next(
        remaining_records,
        effective_target,
        minimum_records,
        maximum_records,
        sequence,
        seed,
    )
}

#[cfg(test)]
mod tests {
    use super::next;

    #[test]
    fn resource_signals_remain_bounded_and_repeatable() {
        let relaxed = next(100_000, 1_000, 128, 2_000, 7, 99, 0, 0);
        let pressured = next(100_000, 1_000, 128, 2_000, 7, 99, 100, 0);
        let aggressive = next(100_000, 1_000, 128, 2_000, 7, 99, 0, 100);
        assert_eq!(pressured, next(100_000, 1_000, 128, 2_000, 7, 99, 100, 0));
        assert!((128..=2_000).contains(&relaxed));
        assert!((128..=2_000).contains(&pressured));
        assert!((128..=2_000).contains(&aggressive));
    }

    #[test]
    fn tail_is_exact_even_when_smaller_than_minimum() {
        assert_eq!(next(31, 1_000, 128, 2_000, 0, 0, 100, 0), 31);
    }
}
