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
  uint32_t memory_pressure_percent;
  uint32_t aggression_percent;
} kr_c_chunk_cursor;

enum {
  KR_C_EXECUTION_AUTO = 0,
  KR_C_EXECUTION_CPU = 1,
  KR_C_EXECUTION_HYBRID = 2
};
enum {
  KR_C_PACK_CAREFUL = 0,
  KR_C_PACK_BALANCED = 1,
  KR_C_PACK_THROUGHPUT = 2,
  KR_C_PACK_MAXIMUM = 3
};
enum {
  KR_C_POLICY_OK = 0,
  KR_C_POLICY_INVALID_ARGUMENT = 1,
  KR_C_POLICY_INCOMPATIBLE_ABI = 2,
  KR_C_POLICY_UNAVAILABLE_ACCELERATOR = 3
};
enum {
  KR_C_PLAN_CPU = 1U,
  KR_C_PLAN_HYBRID = 2U,
  KR_C_PLAN_EXPLICIT_THREADS = 4U
};

typedef struct kr_c_execution_request {
  uint32_t abi_version;
  uint32_t execution;
  uint32_t algorithm_pack;
  uint32_t cpu_percent;
  uint32_t ram_percent;
  uint32_t gpu_percent;
  uint32_t requested_threads;
  uint32_t logical_cpu_threads;
  uint32_t accelerator_available;
  uint64_t available_memory_bytes;
  uint64_t dataset_records;
  size_t record_bytes;
} kr_c_execution_request;

typedef struct kr_c_execution_plan {
  uint32_t flags;
  uint32_t native_threads;
  uint32_t data_workers;
  uint32_t prefetch;
  uint32_t memory_pressure_percent;
  uint32_t aggression_percent;
  size_t target_records;
  size_t minimum_records;
  size_t maximum_records;
} kr_c_execution_plan;

int kr_c_chunk_cursor_init(
    kr_c_chunk_cursor* cursor,
    size_t records,
    size_t target_records,
    size_t minimum_records,
    size_t maximum_records,
    uint64_t seed);
size_t kr_c_chunk_cursor_next(kr_c_chunk_cursor* cursor);
int kr_c_chunk_cursor_init_from_plan(
    kr_c_chunk_cursor* cursor,
    size_t records,
    const kr_c_execution_plan* plan,
    uint64_t seed);
int kr_c_context_split(uint64_t context_key, uint32_t validation_percent, uint32_t test_percent);
int kr_c_execution_plan_make(
    const kr_c_execution_request* request,
    kr_c_execution_plan* plan);

#ifdef __cplusplus
}
#endif
