#pragma once

#include <stddef.h>
#include <stdint.h>

#include "kernelyra_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kr_c_chunk_cursor {
  size_t remaining_records;
  size_t target_records;
  size_t minimum_records;
  size_t maximum_records;
  uint64_t sequence;
  uint64_t seed;
} kr_c_chunk_cursor;

enum { KR_C_POLICY_OK = 0, KR_C_POLICY_INVALID_ARGUMENT = 1 };

int kr_c_chunk_cursor_init(
    kr_c_chunk_cursor* cursor,
    size_t records,
    size_t target_records,
    size_t minimum_records,
    size_t maximum_records,
    uint64_t seed);
size_t kr_c_chunk_cursor_next(kr_c_chunk_cursor* cursor);
int kr_c_context_split(uint64_t context_key, uint32_t validation_percent, uint32_t test_percent);

#ifdef __cplusplus
}
#endif
