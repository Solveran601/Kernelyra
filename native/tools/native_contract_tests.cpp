#include "kernelyra_core.h"
#include "kernelyra_policy.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const char* message) {
  if (condition) return true;
  std::cerr << "native contract test failed: " << message << '\n';
  return false;
}

}  // namespace

int main() {
  float values[] = {1.0F, std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity(), -2.0F,
                    10.0F, -10.0F};
  const float means[] = {0.0F, 2.0F};
  const float stds[] = {1.0F, 2.0F};
  uint64_t repaired = 0U;
  const uint32_t preprocessing = KR_PREPROCESS_IMPUTE_NONFINITE |
                                 KR_PREPROCESS_NORMALIZE |
                                 KR_PREPROCESS_CLIP;
  if (!expect(kr_preprocess_f32(values, 3U, 2U, means, stds, 3.0F, preprocessing, &repaired) == 1,
              "fused preprocessing succeeds") ||
      !expect(repaired == 2U, "non-finite values are repaired") ||
      !expect(kr_values_all_finite_f32(values, 6U) == 1U, "preprocessing returns finite values") ||
      !expect(std::abs(values[0] - 1.0F) < 1.0e-6F && std::abs(values[1]) < 1.0e-6F,
              "normalization keeps expected values") ||
      !expect(values[4] == 3.0F && values[5] == -3.0F, "clipping is fused")) {
    return 1;
  }
  const std::string text = "First sentence is complete. Second sentence keeps the meaning.\n\nThird paragraph ends the example.";
  const size_t chunk_count = kr_text_plan_chunks(
      reinterpret_cast<const uint8_t*>(text.data()), text.size(), 24U, 42U, 56U, 12U, nullptr, 0U);
  if (!expect(chunk_count != KR_TEXT_CHUNK_PLAN_INVALID && chunk_count >= 2U,
              "text planner creates multiple chunks")) {
    return 1;
  }
  std::vector<kr_text_chunk> text_chunks(chunk_count);
  if (!expect(kr_text_plan_chunks(
                  reinterpret_cast<const uint8_t*>(text.data()), text.size(), 24U, 42U, 56U, 12U,
                  text_chunks.data(), text_chunks.size()) == chunk_count,
              "text planner writes the announced span count")) {
    return 1;
  }
  size_t covered = 0U;
  for (const kr_text_chunk& chunk : text_chunks) {
    if (!expect(chunk.context_start <= chunk.content_start && chunk.content_start == covered &&
                    chunk.end > chunk.content_start && chunk.end <= text.size(),
                "text chunks retain contiguous content and bounded context")) {
      return 1;
    }
    covered = chunk.end;
  }
  if (!expect(covered == text.size(), "text chunks cover the complete source")) return 1;
  float unrepaired[] = {std::numeric_limits<float>::quiet_NaN()};
  if (!expect(kr_preprocess_f32(
                  unrepaired, 1U, 1U, means, stds, 3.0F,
                  KR_PREPROCESS_NORMALIZE | KR_PREPROCESS_CLIP, nullptr) == 0,
              "non-finite values are rejected when repair is disabled")) {
    return 1;
  }

  kr_c_execution_request request{};
  request.abi_version = KR_ABI_VERSION;
  request.execution = KR_C_EXECUTION_CPU;
  request.algorithm_pack = KR_C_PACK_THROUGHPUT;
  request.cpu_percent = 90U;
  request.ram_percent = 75U;
  request.gpu_percent = 0U;
  request.logical_cpu_threads = 16U;
  request.available_memory_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  request.dataset_records = 100000U;
  request.record_bytes = 128U;
  kr_c_execution_plan plan{};
  if (!expect(kr_c_execution_plan_make(&request, &plan) == KR_C_POLICY_OK, "C plan is valid") ||
      !expect(plan.native_threads >= 1U && plan.native_threads <= 16U, "thread cap is respected") ||
      !expect(plan.minimum_records <= plan.maximum_records && plan.target_records > 0U,
              "chunk bounds are valid")) {
    return 1;
  }
  request.gpu_percent = 1U;
  if (!expect(kr_c_execution_plan_make(&request, &plan) == KR_C_POLICY_INVALID_ARGUMENT,
              "CPU execution rejects a GPU reservation")) {
    return 1;
  }
  request.execution = KR_C_EXECUTION_HYBRID;
  if (!expect(kr_c_execution_plan_make(&request, &plan) == KR_C_POLICY_UNAVAILABLE_ACCELERATOR,
              "hybrid execution requires an available accelerator")) {
    return 1;
  }
  request.execution = KR_C_EXECUTION_CPU;
  request.gpu_percent = 0U;
  if (!expect(kr_c_execution_plan_make(&request, &plan) == KR_C_POLICY_OK,
              "validated request remains reusable")) {
    return 1;
  }
  kr_c_chunk_cursor cursor{};
  if (!expect(kr_c_chunk_cursor_init_from_plan(&cursor, 100000U, &plan, 20260826U) == KR_C_POLICY_OK,
              "C cursor accepts the plan")) {
    return 1;
  }
  size_t scheduled = 0U;
  while (cursor.remaining_records != 0U) {
    const size_t chunk = kr_c_chunk_cursor_next(&cursor);
    if (!expect(chunk > 0U && chunk <= 100000U - scheduled, "cursor remains bounded")) return 1;
    scheduled += chunk;
  }
  if (!expect(scheduled == 100000U, "cursor schedules every record once")) return 1;
  return 0;
}
