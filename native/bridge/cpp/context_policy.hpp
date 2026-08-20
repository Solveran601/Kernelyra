#pragma once

#include <cstddef>
#include <cstdint>

namespace kernelyra::policy {

uint64_t mix_u64(uint64_t value);
uint32_t split_for_context(uint64_t context_key, uint32_t validation_percent, uint32_t test_percent);
size_t next_chunk_size(
    size_t remaining_records,
    size_t target_records,
    size_t minimum_records,
    size_t maximum_records,
    uint64_t sequence,
    uint64_t seed);

}  // namespace kernelyra::policy
