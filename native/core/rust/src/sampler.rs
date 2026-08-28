//! Allocation-free xorshift64* sampling shared with the C++ training bridge.

#[inline]
fn next_random(state: &mut u64) -> u64 {
    let mut value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    value.wrapping_mul(2_685_821_657_736_338_717)
}

pub fn fill_indices(rows: usize, state: &mut u64, output: &mut [usize]) {
    for index in output {
        *index = (next_random(state) as usize) % rows;
    }
}
