//! Allocation-free, deterministic policies shared by the Kernelyra native core.
//!
//! Rust owns policy arithmetic that must be identical across every caller. C++
//! owns streaming, Fortran owns dense arithmetic and Zig owns buffer work.

// Export names are explicitly marked unsafe by modern Rust because the linker
// namespace is global.  This crate contains no unsafe operations or pointers.
#![deny(unsafe_op_in_unsafe_fn)]

mod chunks;
mod hash;
mod signature;
mod split;

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
    fn signatures_are_classified_from_a_bounded_prefix() {
        assert_eq!(signature::classify(b"PAR1schema"), signature::PARQUET);
        assert_eq!(signature::classify(b"SQLite format 3\0"), signature::SQLITE);
        assert_eq!(signature::classify(b"\x89PNG\r\n\x1a\npixels"), signature::PNG);
        assert_eq!(signature::classify(b"a,b\n1,2\n"), signature::DELIMITED_TEXT);
        assert_eq!(signature::classify(&[0; 32]), signature::UNKNOWN);
    }
}
