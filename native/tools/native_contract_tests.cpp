#include "kernelyra_core.h"
#include "kernelyra_policy.h"
#include "execution_guard.h"

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
  float zig_values[] = {1.0F, -2.0F, 3.0F};
  const float zig_add[] = {1.0F, 2.0F, 3.0F};
  kr_memory_fill_f32(zig_values, 3U, 2.0F);
  kr_memory_scale_f32(zig_values, 3U, 0.5F);
  kr_memory_add_f32(zig_values, zig_add, 3U);
  if (!expect(std::abs(kr_values_sum_f32(zig_values, 3U) - 9.0F) < 1.0e-6F,
              "Zig sum uses the transformed buffer") ||
      !expect(std::abs(kr_values_max_abs_f32(zig_values, 3U) - 4.0F) < 1.0e-6F,
              "Zig max-absolute reduction is correct")) {
    return 1;
  }
  float repair_only[] = {std::numeric_limits<float>::quiet_NaN(), 3.0F,
                         std::numeric_limits<float>::infinity(), -1.0F};
  if (!expect(kr_memory_repair_nonfinite_f32(repair_only, 2U, 2U, means) == 2U,
              "Zig repair counts non-finite matrix values") ||
      !expect(repair_only[0] == 0.0F && repair_only[2] == 0.0F,
              "Zig repair uses the matching feature mean")) {
    return 1;
  }
  void* arena = kr_memory_arena_create(1024U, 64U);
  if (!expect(arena != nullptr, "Zig monotonic arena is created")) return 1;
  const uintptr_t first_address = reinterpret_cast<uintptr_t>(kr_memory_arena_acquire(arena, 96U, 64U));
  const size_t mark = kr_memory_arena_mark(arena);
  const uintptr_t second_address = reinterpret_cast<uintptr_t>(kr_memory_arena_acquire(arena, 128U, 64U));
  kr_memory_arena_stats arena_stats{};
  const bool arena_valid = first_address != 0U && second_address != 0U &&
                           first_address % 64U == 0U && second_address % 64U == 0U &&
                           kr_memory_arena_get_stats(arena, &arena_stats) == 1 &&
                           arena_stats.capacity_bytes == 1024U && arena_stats.used_bytes > mark &&
                           arena_stats.allocations == 2U;
  if (!expect(arena_valid, "Zig arena returns aligned monotonic leases and records statistics") ||
      !expect(kr_memory_arena_rewind(arena, mark) == 1, "Zig arena rewinds to a validated mark") ||
      !expect(kr_memory_arena_acquire(arena, 2048U, 64U) == nullptr,
              "Zig arena rejects a lease beyond its hard capacity") ||
      !expect(kr_memory_arena_get_stats(arena, &arena_stats) == 1 && arena_stats.failed_allocations == 1U,
              "Zig arena records rejected leases")) {
    kr_memory_arena_destroy(arena);
    return 1;
  }
  kr_memory_arena_reset(arena);
  if (!expect(kr_memory_arena_get_stats(arena, &arena_stats) == 1 && arena_stats.used_bytes == 0U &&
                  arena_stats.resets == 1U,
              "Zig arena resets leases in constant time")) {
    kr_memory_arena_destroy(arena);
    return 1;
  }
  kr_memory_arena_destroy(arena);
  kr_memory_batch_plan batch_plan{};
  if (!expect(kr_memory_batch_plan_make(4096U, 200U, 3U, 2U, &batch_plan) == 1,
              "Zig batch planner accepts a fitting aligned feature and target pair") ||
      !expect(batch_plan.fits == 1U && batch_plan.planned_rows == 200U && batch_plan.bytes_per_row == 16U &&
                  batch_plan.required_bytes == 3264U,
              "Zig batch planner accounts for float32 values and alignment reserve") ||
      !expect(kr_memory_batch_plan_make(4096U, 300U, 3U, 2U, &batch_plan) == 1 && batch_plan.fits == 0U &&
                  batch_plan.maximum_rows == 252U && batch_plan.planned_rows == 252U,
              "Zig batch planner reports the maximum safe row count without allocating")) {
    return 1;
  }
  const float moment_values[] = {1.0F, 2.0F, 3.0F, 4.0F};
  float mean = 0.0F;
  float standard_deviation = 0.0F;
  if (!expect(kr_values_moments_f32(moment_values, 4U, &mean, &standard_deviation) == 1,
              "Fortran moments accepts a finite vector") ||
      !expect(std::abs(mean - 2.5F) < 1.0e-6F && std::abs(standard_deviation - std::sqrt(1.25F)) < 1.0e-6F,
              "Fortran moments are stable and correct")) {
    return 1;
  }
  float logits[] = {-2.0F, 0.0F, 1.0F};
  if (!expect(kr_values_softmax_f32(logits, 3U) == 1, "Fortran softmax succeeds") ||
      !expect(std::abs(kr_values_sum_f32(logits, 3U) - 1.0F) < 1.0e-5F && logits[2] > logits[1] && logits[1] > logits[0],
              "Fortran softmax returns an ordered probability distribution")) {
    return 1;
  }
  float gradient[] = {3.0F, 4.0F};
  if (!expect(std::abs(kr_values_clip_l2_f32(gradient, 2U, 2.0F) - 5.0F) < 1.0e-6F,
              "Fortran gradient clip reports the observed norm") ||
      !expect(std::abs(kr_values_l2_norm_f32(gradient, 2U) - 2.0F) < 1.0e-5F,
              "Fortran gradient clip enforces the requested L2 bound")) {
    return 1;
  }

  kr_c_batch_contract core_batch{};
  if (!expect(kr_c_core_batch_contract_make(12U, 8U, 4U, &core_batch) == 1,
              "C core validates a bounded random minibatch") ||
      !expect(core_batch.dataset_elements == 96U && core_batch.batch_elements == 32U &&
                  core_batch.dataset_bytes == 384U && core_batch.batch_bytes == 128U,
              "C core publishes exact float32 storage requirements") ||
      !expect(kr_c_core_batch_contract_make(12U, 8U, 13U, &core_batch) == 0,
              "C core rejects a batch that exceeds its source table")) {
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
  const float multiclass_x[] = {
      -2.0F, -1.0F, -1.5F, -2.0F,
       2.0F, -1.0F,  1.5F, -2.0F,
       0.0F,  2.0F,  0.5F,  1.5F,
  };
  const float multiclass_y[] = {0.0F, 0.0F, 1.0F, 1.0F, 2.0F, 2.0F};
  kr_model_config multiclass{};
  multiclass.abi_version = KR_ABI_VERSION;
  multiclass.task = KR_TASK_MULTICLASS;
  multiclass.features = 2U;
  multiclass.classes = 3U;
  multiclass.learning_rate = 0.1F;
  multiclass.weight_decay = 0.0F;
  void* model = kr_model_create(&multiclass);
  if (!expect(model != nullptr, "multiclass model is created")) return 1;
  float first_loss = 0.0F;
  float final_loss = 0.0F;
  for (size_t step = 0U; step < 80U; ++step) {
    float* destination = step == 0U ? &first_loss : &final_loss;
    if (!expect(kr_model_train_step(model, multiclass_x, multiclass_y, 6U, destination) == 1,
                "Fortran multiclass training step succeeds")) {
      kr_model_destroy(model);
      return 1;
    }
  }
  std::vector<float> probabilities(18U);
  const bool prediction_ok = kr_model_predict(model, multiclass_x, 6U, probabilities.data(), probabilities.size()) == 1;
  kr_model_destroy(model);
  if (!expect(prediction_ok && std::isfinite(final_loss) && final_loss < first_loss,
              "Fortran multiclass core reduces loss and predicts")) {
    return 1;
  }

  const float binary_x[] = {
      -2.0F, -1.0F, -1.0F, -1.5F, -1.0F, -0.5F,
       1.0F,  0.5F,  1.0F,  1.5F,  2.0F,  1.0F,
  };
  const float binary_y[] = {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
  kr_model_config binary{};
  binary.abi_version = KR_ABI_VERSION;
  binary.task = KR_TASK_BINARY;
  binary.features = 2U;
  binary.classes = 1U;
  binary.learning_rate = 0.1F;
  void* binary_model = kr_model_create(&binary);
  if (!expect(binary_model != nullptr, "binary model is created")) return 1;
  first_loss = 0.0F;
  final_loss = 0.0F;
  if (!expect(kr_model_train_step(binary_model, binary_x, binary_y, 6U, &first_loss) == 1 &&
                  kr_model_train_steps(binary_model, binary_x, binary_y, 6U, 79U, &final_loss) == 1,
              "tiled Fortran binary single and bulk training succeed")) {
    kr_model_destroy(binary_model);
    return 1;
  }
  std::vector<float> binary_probabilities(6U);
  const bool binary_prediction_ok =
      kr_model_predict(binary_model, binary_x, 6U, binary_probabilities.data(), binary_probabilities.size()) == 1;
  float random_loss = 0.0F;
  const bool random_batch_ok =
      kr_model_train_random_step(binary_model, binary_x, binary_y, 6U, 4U, &random_loss) == 1;
  const uint32_t random_execution = kr_model_execution_mask(binary_model);
  kr_model_destroy(binary_model);
  if (!expect(binary_prediction_ok && std::isfinite(final_loss) && final_loss < first_loss &&
                  binary_probabilities.front() < 0.5F && binary_probabilities.back() > 0.5F,
              "tiled Fortran binary core learns a separable batch")) {
    return 1;
  }
  if (!expect(random_batch_ok && std::isfinite(random_loss) &&
                  (random_execution & (KR_EXECUTION_C_ABI | KR_EXECUTION_CPP_DISPATCH |
                                       KR_EXECUTION_RUST_POLICY | KR_EXECUTION_FORTRAN_NUMERIC |
                                       KR_EXECUTION_ZIG_MEMORY)) ==
                      (KR_EXECUTION_C_ABI | KR_EXECUTION_CPP_DISPATCH |
                       KR_EXECUTION_RUST_POLICY | KR_EXECUTION_FORTRAN_NUMERIC |
                       KR_EXECUTION_ZIG_MEMORY),
              "random batch composes C, C++, Rust, Zig and Fortran")) {
    return 1;
  }
  void* guarded_binary_model = kr_model_create(&binary);
  if (!expect(guarded_binary_model != nullptr, "guarded binary model is created")) return 1;
  std::vector<float> weights_before(2U);
  std::vector<float> bias_before(1U);
  std::vector<float> weights_after(2U);
  std::vector<float> bias_after(1U);
  if (!expect(kr_model_export(guarded_binary_model, weights_before.data(), weights_before.size(),
                              bias_before.data(), bias_before.size()) == 1,
              "guarded binary model is exportable")) {
    kr_model_destroy(guarded_binary_model);
    return 1;
  }
  std::vector<float> nonfinite_binary_x(binary_x, binary_x + 12U);
  nonfinite_binary_x[0] = std::numeric_limits<float>::quiet_NaN();
  float guarded_loss = 0.0F;
  const int guarded_update =
      kr_model_train_step(guarded_binary_model, nonfinite_binary_x.data(), binary_y, 6U, &guarded_loss);
  const int guarded_export = kr_model_export(guarded_binary_model, weights_after.data(), weights_after.size(),
                                             bias_after.data(), bias_after.size());
  kr_model_destroy(guarded_binary_model);
  if (!expect(guarded_update == 0 && guarded_export == 1 && weights_after == weights_before &&
                  bias_after == bias_before,
              "Fortran pre-update guard rejects non-finite batches without changing parameters")) {
    return 1;
  }

  kr_model_config overflow_guard{};
  overflow_guard.abi_version = KR_ABI_VERSION;
  overflow_guard.task = KR_TASK_BINARY;
  overflow_guard.features = 1U;
  overflow_guard.classes = 1U;
  overflow_guard.learning_rate = std::numeric_limits<float>::max();
  void* overflow_guard_model = kr_model_create(&overflow_guard);
  if (!expect(overflow_guard_model != nullptr, "overflow-guard model is created")) return 1;
  float overflow_weight_before = 0.0F;
  float overflow_bias_before = 0.0F;
  if (!expect(kr_model_export(
                  overflow_guard_model, &overflow_weight_before, 1U, &overflow_bias_before, 1U) == 1,
              "overflow-guard model is exportable")) {
    kr_model_destroy(overflow_guard_model);
    return 1;
  }
  const float overflow_x[] = {100.0F};
  const float overflow_y[] = {
      (overflow_weight_before * overflow_x[0] + overflow_bias_before) >= 0.0F ? 0.0F : 1.0F};
  float overflow_loss = 0.0F;
  float overflow_weight_after = 0.0F;
  float overflow_bias_after = 0.0F;
  const bool overflow_rejected =
      kr_model_train_step(overflow_guard_model, overflow_x, overflow_y, 1U, &overflow_loss) == 0 &&
      kr_model_export(
          overflow_guard_model, &overflow_weight_after, 1U, &overflow_bias_after, 1U) == 1;
  kr_model_destroy(overflow_guard_model);
  if (!expect(overflow_rejected && overflow_weight_before == overflow_weight_after &&
                  overflow_bias_before == overflow_bias_after,
              "Fortran preflight rejects float32-overflowing updates without mutating parameters")) {
    return 1;
  }

  const float regression_x[] = {-3.0F, -2.0F, -1.0F, 1.0F, 2.0F, 3.0F};
  const float regression_y[] = {-6.0F, -4.0F, -2.0F, 2.0F, 4.0F, 6.0F};
  kr_model_config regression{};
  regression.abi_version = KR_ABI_VERSION;
  regression.task = KR_TASK_REGRESSION;
  regression.features = 1U;
  regression.classes = 1U;
  regression.learning_rate = 0.04F;
  regression.target_mean = 0.0F;
  regression.target_std = 1.0F;
  void* regression_model = kr_model_create(&regression);
  if (!expect(regression_model != nullptr, "regression model is created")) return 1;
  first_loss = 0.0F;
  final_loss = 0.0F;
  if (!expect(kr_model_train_step(regression_model, regression_x, regression_y, 6U, &first_loss) == 1 &&
                  kr_model_train_steps(
                      regression_model, regression_x, regression_y, 6U, 99U, &final_loss) == 1,
              "tiled Fortran regression single and bulk training succeed")) {
    kr_model_destroy(regression_model);
    return 1;
  }
  kr_model_destroy(regression_model);
  if (!expect(std::isfinite(final_loss) && final_loss < first_loss,
              "tiled Fortran regression core reduces loss")) {
    return 1;
  }
  return 0;
}
