#include "kernelyra_policy.h"

#include <limits.h>

int kr_c_chunk_cursor_init(
    kr_c_chunk_cursor* cursor,
    size_t records,
    size_t target_records,
    size_t minimum_records,
    size_t maximum_records,
    uint64_t seed) {
  if (cursor == NULL || target_records == 0U || minimum_records == 0U || maximum_records < minimum_records) {
    return KR_C_POLICY_INVALID_ARGUMENT;
  }
  cursor->remaining_records = records;
  cursor->target_records = target_records;
  cursor->minimum_records = minimum_records;
  cursor->maximum_records = maximum_records;
  cursor->sequence = 0U;
  cursor->seed = seed;
  return KR_C_POLICY_OK;
}

size_t kr_c_chunk_cursor_next(kr_c_chunk_cursor* cursor) {
  size_t chunk;
  if (cursor == NULL || cursor->remaining_records == 0U) return 0U;
  chunk = kr_rust_next_chunk_size(
      cursor->remaining_records,
      cursor->target_records,
      cursor->minimum_records,
      cursor->maximum_records,
      cursor->sequence,
      cursor->seed);
  if (chunk == 0U || chunk > cursor->remaining_records) return 0U;
  cursor->remaining_records -= chunk;
  cursor->sequence += 1U;
  return chunk;
}

int kr_c_context_split(uint64_t context_key, uint32_t validation_percent, uint32_t test_percent) {
  const uint32_t split = kr_rust_split_for_key(context_key, validation_percent, test_percent);
  return split <= KR_SPLIT_TEST ? (int)split : -1;
}
