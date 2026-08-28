//! Allocation-free, deterministic policies shared by the Kernelyra native core.
//!
//! Rust owns policy arithmetic that must be identical across every caller. C++
//! owns streaming, Fortran owns dense arithmetic and Zig owns buffer work.

// Export names are explicitly marked unsafe by modern Rust because the linker
// namespace is global.  This crate contains no unsafe operations or pointers.
#![deny(unsafe_op_in_unsafe_fn)]

mod adaptive;
mod batch_plan;
mod chunks;
mod hash;
mod signature;
mod sampler;
mod split;
mod text_chunks;

#[unsafe(export_name = "kr_rust_policy_mix_u64")]
pub extern "C" fn kr_rust_mix_u64(value: u64) -> u64 {
    hash::mix_u64(value)
}

/// Assign one complete context group to a split without storing a row index.
#[unsafe(export_name = "kr_rust_policy_split_for_key")]
pub extern "C" fn kr_rust_split_for_key(
    group_key: u64,
    validation_percent: u32,
    test_percent: u32,
) -> u32 {
    split::for_context(group_key, validation_percent, test_percent)
}

/// Return a bounded variable chunk size for context-safe stream scheduling.
#[unsafe(export_name = "kr_rust_policy_next_chunk_size")]
pub extern "C" fn kr_rust_next_chunk_size(
    remaining_records: usize,
    target_records: usize,
    minimum_records: usize,
    maximum_records: usize,
    sequence: u64,
    seed: u64,
) -> usize {
    chunks::next(
        remaining_records,
        target_records,
        minimum_records,
        maximum_records,
        sequence,
        seed,
    )
}

/// Resource-aware deterministic chunk policy used by the C execution planner.
#[unsafe(export_name = "kr_rust_policy_next_adaptive_chunk_size")]
pub extern "C" fn kr_rust_next_adaptive_chunk_size(
    remaining_records: usize,
    target_records: usize,
    minimum_records: usize,
    maximum_records: usize,
    sequence: u64,
    seed: u64,
    memory_pressure_percent: u32,
    aggression_percent: u32,
) -> usize {
    adaptive::next(
        remaining_records,
        target_records,
        minimum_records,
        maximum_records,
        sequence,
        seed,
        memory_pressure_percent,
        aggression_percent,
    )
}

/// Fill a native mini-batch index buffer without allocating. The C++ bridge
/// passes its persistent RNG state, so sampling remains deterministic across
/// Python, C and C++ callers and does not create a Python-side index array.
///
/// # Safety
/// `state` and `output` must be valid writable pointers. `output` must point
/// to at least `requested` `usize` elements.
#[unsafe(export_name = "kr_rust_policy_sample_indices")]
pub unsafe extern "C" fn kr_rust_sample_indices(
    rows: usize,
    requested: usize,
    state: *mut u64,
    output: *mut usize,
) -> usize {
    if rows == 0 || requested == 0 || state.is_null() || output.is_null() {
        return 0;
    }
    let rng_state = unsafe { &mut *state };
    let destination = unsafe { core::slice::from_raw_parts_mut(output, requested) };
    sampler::fill_indices(rows, rng_state, destination);
    requested
}

/// Validate native minibatch dimensions without allocating. The bridge uses
/// this before resizing persistent C++ buffers, so a malformed request cannot
/// reach pointer arithmetic in the gathering kernels.
///
/// # Safety
/// `output_values` must point to writable `usize` storage.
#[unsafe(export_name = "kr_rust_policy_plan_batch")]
pub unsafe extern "C" fn kr_rust_plan_batch(
    source_rows: usize,
    features: usize,
    requested_rows: usize,
    output_values: *mut usize,
    output_bytes: *mut usize,
) -> usize {
    if output_values.is_null() || output_bytes.is_null() {
        return 0;
    }
    let Some(plan) = batch_plan::plan(source_rows, features, requested_rows) else {
        return 0;
    };
    unsafe {
        *output_values = plan.values;
        *output_bytes = plan.values_bytes;
    }
    plan.rows
}

/// Classify an untrusted file prefix without parsing or allocating from it.
///
/// # Safety
/// `bytes` must either be null with a zero length, or reference at least
/// `length` readable bytes. Only the first 4096 bytes are examined.
#[unsafe(export_name = "kr_rust_policy_probe_signature")]
pub unsafe extern "C" fn kr_rust_probe_signature(bytes: *const u8, length: usize) -> u32 {
    if bytes.is_null() || length == 0 {
        return signature::UNKNOWN;
    }
    let bounded = length.min(4096);
    let prefix = unsafe { core::slice::from_raw_parts(bytes, bounded) };
    signature::classify(prefix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_is_stable_and_leaves_training_capacity() {
        assert_eq!(kr_rust_split_for_key(42, 15, 15), kr_rust_split_for_key(42, 15, 15));
        assert_eq!(kr_rust_split_for_key(42, 80, 16), split::INVALID);
        assert!((0..10_000).any(|key| kr_rust_split_for_key(key, 15, 15) == split::TRAIN));
    }

    #[test]
    fn chunks_are_bounded_and_finish_with_a_short_tail() {
        assert!((256..=768).contains(&kr_rust_next_chunk_size(10_000, 512, 256, 768, 3, 99)));
        assert_eq!(kr_rust_next_chunk_size(19, 512, 256, 768, 3, 99), 19);
        assert_eq!(kr_rust_next_chunk_size(10, 0, 1, 2, 0, 0), 0);
    }

    #[test]
    fn adaptive_chunks_are_repeatable() {
        let left = kr_rust_next_adaptive_chunk_size(50_000, 1_024, 256, 2_048, 3, 99, 45, 70);
        let right = kr_rust_next_adaptive_chunk_size(50_000, 1_024, 256, 2_048, 3, 99, 45, 70);
        assert_eq!(left, right);
        assert!((256..=2_048).contains(&left));
    }

    #[test]
    fn signatures_are_classified_from_a_bounded_prefix() {
        assert_eq!(signature::classify(b"PAR1schema"), signature::PARQUET);
        assert_eq!(signature::classify(b"SQLite format 3\0"), signature::SQLITE);
        assert_eq!(signature::classify(b"\x89PNG\r\n\x1a\npixels"), signature::PNG);
        assert_eq!(signature::classify(b"a,b\n1,2\n"), signature::DELIMITED_TEXT);
        assert_eq!(signature::classify(&[0; 32]), signature::UNKNOWN);
    }

    #[test]
    fn sampler_is_deterministic_and_bounded() {
        let mut first_state = 7;
        let mut second_state = 7;
        let mut first = [0_usize; 32];
        let mut second = [0_usize; 32];
        sampler::fill_indices(17, &mut first_state, &mut first);
        sampler::fill_indices(17, &mut second_state, &mut second);
        assert_eq!(first, second);
        assert!(first.iter().all(|index| *index < 17));
    }
}
