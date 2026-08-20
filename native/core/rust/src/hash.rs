pub const GOLDEN_RATIO: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_B: u64 = 0x94D0_49BB_1331_11EB;

/// Stable SplitMix64 finalizer used for row and context keys.
pub fn mix_u64(value: u64) -> u64 {
    let value = value.wrapping_add(GOLDEN_RATIO);
    let value = (value ^ (value >> 30)).wrapping_mul(MIX_A);
    let value = (value ^ (value >> 27)).wrapping_mul(MIX_B);
    value ^ (value >> 31)
}
