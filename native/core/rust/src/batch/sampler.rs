//! Allocation-free xorshift64* sampling shared with the C++ training bridge.

const NONZERO_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

#[inline(always)]
fn next_random(state: &mut u64) -> u64 {
    let mut value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    value.wrapping_mul(2_685_821_657_736_338_717)
}

pub fn fill_indices(rows: usize, state: &mut u64, output: &mut [usize]) {
    // xorshift's all-zero state would otherwise repeat index zero forever.
    // C++ model construction already uses a non-zero seed; this also keeps
    // direct FFI callers statistically useful without allocating.
    if *state == 0 {
        *state = NONZERO_SEED;
    }
    for index in output {
        *index = (next_random(state) as usize) % rows;
    }
}
