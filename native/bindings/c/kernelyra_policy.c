#include "kernelyra_policy.h"

#include <limits.h>

static size_t kr_c_min_size(size_t left, size_t right) { return left < right ? left : right; }

static size_t kr_c_max_size(size_t left, size_t right) { return left > right ? left : right; }

static size_t kr_c_scale_size(size_t value, uint32_t percent) {
  if (value == 0U || percent == 0U) return 0U;
  if (value > SIZE_MAX / percent) return SIZE_MAX;
  return value * percent / 100U;
}

static size_t kr_c_records_from_memory(uint64_t bytes, uint32_t ram_percent, size_t record_bytes, uint32_t reserve) {
  uint64_t usable;
  uint64_t records;
  if (bytes == 0U || ram_percent == 0U || record_bytes == 0U || reserve == 0U) return 0U;
  usable = bytes / reserve;
  /* Divide before multiplying so an extreme host-memory value cannot wrap. */
  usable = (usable / 100U) * ram_percent + (usable % 100U) * ram_percent / 100U;
  records = usable / record_bytes;
  return records > SIZE_MAX ? SIZE_MAX : (size_t)records;
}

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
  cursor->memory_pressure_percent = 0U;
  cursor->aggression_percent = 0U;
  return KR_C_POLICY_OK;
}

size_t kr_c_chunk_cursor_next(kr_c_chunk_cursor* cursor) {
  size_t chunk;
  if (cursor == NULL || cursor->remaining_records == 0U) return 0U;
  chunk = kr_rust_next_adaptive_chunk_size(
      cursor->remaining_records,
      cursor->target_records,
      cursor->minimum_records,
      cursor->maximum_records,
      cursor->sequence,
      cursor->seed,
      cursor->memory_pressure_percent,
      cursor->aggression_percent);
  if (chunk == 0U || chunk > cursor->remaining_records) return 0U;
  cursor->remaining_records -= chunk;
  cursor->sequence += 1U;
  return chunk;
}

int kr_c_context_split(uint64_t context_key, uint32_t validation_percent, uint32_t test_percent) {
  const uint32_t split = kr_rust_split_for_key(context_key, validation_percent, test_percent);
  return split <= KR_SPLIT_TEST ? (int)split : -1;
}

int kr_c_context_split_seeded(
    uint64_t context_key, uint64_t seed, uint32_t validation_percent, uint32_t test_percent) {
  const uint32_t split =
      kr_rust_split_for_key_seeded(context_key, seed, validation_percent, test_percent);
  return split <= KR_SPLIT_TEST ? (int)split : -1;
}

int kr_c_chunk_cursor_init_from_plan(
    kr_c_chunk_cursor* cursor,
    size_t records,
    const kr_c_execution_plan* plan,
    uint64_t seed) {
  int status;
  if (plan == NULL) return KR_C_POLICY_INVALID_ARGUMENT;
  status = kr_c_chunk_cursor_init(
      cursor, records, plan->target_records, plan->minimum_records, plan->maximum_records, seed);
  if (status != KR_C_POLICY_OK) return status;
  cursor->memory_pressure_percent = plan->memory_pressure_percent;
  cursor->aggression_percent = plan->aggression_percent;
  return KR_C_POLICY_OK;
}

int kr_c_execution_plan_make(
    const kr_c_execution_request* request,
    kr_c_execution_plan* plan) {
  uint32_t reserve;
  uint32_t minimum_percent;
  uint32_t maximum_percent;
  uint32_t worker_cap;
  size_t target;
  size_t minimum;
  size_t maximum;
  if (request == NULL || plan == NULL || request->abi_version != KR_ABI_VERSION ||
      request->logical_cpu_threads == 0U || request->record_bytes == 0U ||
      request->available_memory_bytes == 0U || request->cpu_percent == 0U ||
      request->cpu_percent > 100U || request->ram_percent == 0U || request->ram_percent > 100U ||
      request->gpu_percent > 100U || request->execution > KR_C_EXECUTION_HYBRID) {
    return request != NULL && request->abi_version != KR_ABI_VERSION
        ? KR_C_POLICY_INCOMPATIBLE_ABI : KR_C_POLICY_INVALID_ARGUMENT;
  }
  if (request->execution == KR_C_EXECUTION_CPU && request->gpu_percent != 0U) return KR_C_POLICY_INVALID_ARGUMENT;
  if (request->execution == KR_C_EXECUTION_HYBRID && request->accelerator_available == 0U) {
    return KR_C_POLICY_UNAVAILABLE_ACCELERATOR;
  }
  if (request->requested_threads > request->logical_cpu_threads) return KR_C_POLICY_INVALID_ARGUMENT;

  /* One automatic policy: the caller owns every resource ceiling. */
  reserve = 10U; minimum_percent = 70U; maximum_percent = 135U; worker_cap = 2U;
  plan->prefetch = 2U; plan->aggression_percent = 35U;
  target = kr_c_records_from_memory(
      request->available_memory_bytes, request->ram_percent, request->record_bytes, reserve);
  if (request->dataset_records != 0U && request->dataset_records < target) {
    target = (size_t)request->dataset_records;
  }
  if (target == 0U) return KR_C_POLICY_INVALID_ARGUMENT;
  minimum = kr_c_max_size(1U, kr_c_scale_size(target, minimum_percent));
  maximum = kr_c_max_size(minimum, kr_c_scale_size(target, maximum_percent));
  if (request->dataset_records != 0U) {
    const size_t records = request->dataset_records > SIZE_MAX ? SIZE_MAX : (size_t)request->dataset_records;
    minimum = kr_c_min_size(minimum, records);
    maximum = kr_c_min_size(maximum, records);
    target = kr_c_min_size(target, records);
  }
  plan->native_threads = request->requested_threads != 0U ? request->requested_threads :
      kr_c_max_size(1U, ((size_t)request->logical_cpu_threads * request->cpu_percent + 99U) / 100U);
  plan->data_workers = plan->native_threads > 1U ?
      (uint32_t)kr_c_min_size((size_t)worker_cap, (size_t)plan->native_threads - 1U) : 0U;
  plan->memory_pressure_percent = 100U - request->ram_percent;
  plan->target_records = target;
  plan->minimum_records = minimum;
  plan->maximum_records = maximum;
  plan->flags = request->execution == KR_C_EXECUTION_HYBRID ? KR_C_PLAN_HYBRID : KR_C_PLAN_CPU;
  if (request->requested_threads != 0U) plan->flags |= KR_C_PLAN_EXPLICIT_THREADS;
  return KR_C_POLICY_OK;
}
