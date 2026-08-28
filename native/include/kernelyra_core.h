#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define KR_API __declspec(dllexport)
#else
#define KR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum { KR_ABI_VERSION = 9 };
enum { KR_TASK_BINARY = 0, KR_TASK_MULTICLASS = 1, KR_TASK_REGRESSION = 2 };
enum { KR_SPLIT_TRAIN = 0, KR_SPLIT_VALIDATION = 1, KR_SPLIT_TEST = 2 };
enum {
  KR_COMPONENT_ZIG_MEMORY = 1U,
  KR_COMPONENT_FORTRAN_NUMERIC = 2U,
  KR_COMPONENT_RUST_POLICY = 4U,
  KR_COMPONENT_ALL = 7U
};
enum {
  KR_PREPROCESS_IMPUTE_NONFINITE = 1U,
  KR_PREPROCESS_NORMALIZE = 2U,
  KR_PREPROCESS_CLIP = 4U
};
enum {
  KR_EXECUTION_C_ABI = 1U,
  KR_EXECUTION_CPP_DISPATCH = 2U,
  KR_EXECUTION_RUST_POLICY = 4U,
  KR_EXECUTION_FORTRAN_NUMERIC = 8U,
  KR_EXECUTION_ZIG_MEMORY = 16U
};

typedef struct kr_model_config {
  uint32_t abi_version;
  uint32_t task;
  uint32_t features;
  uint32_t classes;
  uint32_t threads;
  uint64_t seed;
  float learning_rate;
  float weight_decay;
  float target_mean;
  float target_std;
} kr_model_config;

typedef struct kr_text_chunk {
  size_t context_start;
  size_t content_start;
  size_t end;
} kr_text_chunk;

typedef struct kr_memory_arena_stats {
  size_t capacity_bytes;
  size_t used_bytes;
  size_t high_water_bytes;
  size_t alignment;
  uint64_t allocations;
  uint64_t failed_allocations;
  uint64_t resets;
} kr_memory_arena_stats;

typedef struct kr_memory_batch_plan {
  size_t capacity_bytes;
  size_t requested_rows;
  size_t planned_rows;
  size_t maximum_rows;
  size_t features;
  size_t buffer_count;
  size_t bytes_per_row;
  size_t required_bytes;
  uint32_t fits;
} kr_memory_batch_plan;

#define KR_TEXT_CHUNK_PLAN_INVALID ((size_t)-1)

KR_API const char* kr_core_version(void);
KR_API const char* kr_core_features(void);
KR_API const char* kr_core_components(void);
KR_API uint32_t kr_core_component_mask(void);
KR_API uint32_t kr_core_enabled_component_mask(void);
KR_API uint32_t kr_core_set_component_mask(uint32_t mask);
KR_API const char* kr_last_error(void);
KR_API void* kr_memory_alloc_aligned(size_t bytes, size_t alignment);
KR_API void kr_memory_free_aligned(void* pointer);
/* A single-owner monotonic arena. reset/rewind invalidate released leases. */
KR_API void* kr_memory_arena_create(size_t capacity_bytes, size_t alignment);
KR_API void kr_memory_arena_destroy(void* handle);
KR_API void* kr_memory_arena_acquire(void* handle, size_t bytes, size_t alignment);
KR_API size_t kr_memory_arena_mark(const void* handle);
KR_API int kr_memory_arena_rewind(void* handle, size_t mark);
KR_API void kr_memory_arena_reset(void* handle);
KR_API int kr_memory_arena_get_stats(const void* handle, kr_memory_arena_stats* output);
/* Plan two or more aligned float32 batch buffers without allocating them. */
KR_API int kr_memory_batch_plan_make(
    size_t capacity_bytes,
    size_t requested_rows,
    size_t features,
    size_t buffer_count,
    kr_memory_batch_plan* output);
KR_API void kr_memory_normalize_f32(
    float* data, size_t rows, size_t features, const float* means, const float* stds);
KR_API int kr_preprocess_f32(
    float* data,
    size_t rows,
    size_t features,
    const float* means,
    const float* stds,
    float clip_limit,
    uint32_t flags,
    uint64_t* repaired_values);
KR_API void kr_memory_copy_f32(float* destination, const float* source, size_t values);
KR_API void kr_memory_zero_f32(float* destination, size_t values);
KR_API void kr_memory_fill_f32(float* destination, size_t values, float value);
KR_API void kr_memory_scale_f32(float* destination, size_t values, float scale);
KR_API void kr_memory_add_f32(float* destination, const float* source, size_t values);
KR_API uint64_t kr_memory_repair_nonfinite_f32(
    float* data, size_t rows, size_t features, const float* means);
KR_API uint32_t kr_values_all_finite_f32(const float* values, size_t count);
KR_API float kr_values_l2_norm_f32(const float* values, size_t count);
KR_API void kr_values_clip_f32(float* values, size_t count, float limit);
KR_API float kr_values_clip_l2_f32(float* values, size_t count, float maximum_norm);
KR_API float kr_values_sum_f32(const float* values, size_t count);
KR_API float kr_values_max_abs_f32(const float* values, size_t count);
KR_API int kr_values_moments_f32(
    const float* values, size_t count, float* mean, float* standard_deviation);
KR_API int kr_values_softmax_f32(float* values, size_t count);
KR_API uint64_t kr_rust_mix_u64(uint64_t value);
KR_API uint32_t kr_rust_split_for_key(
    uint64_t group_key, uint32_t validation_percent, uint32_t test_percent);
/* ABI-compatible extension: seed changes a context assignment reproducibly. */
KR_API uint32_t kr_rust_split_for_key_seeded(
    uint64_t group_key, uint64_t seed, uint32_t validation_percent, uint32_t test_percent);
KR_API size_t kr_rust_next_chunk_size(
    size_t remaining_records,
    size_t target_records,
    size_t minimum_records,
    size_t maximum_records,
    uint64_t sequence,
    uint64_t seed);
KR_API size_t kr_rust_next_adaptive_chunk_size(
    size_t remaining_records,
    size_t target_records,
    size_t minimum_records,
    size_t maximum_records,
    uint64_t sequence,
    uint64_t seed,
    uint32_t memory_pressure_percent,
    uint32_t aggression_percent);
/* Plan UTF-8-safe context spans. Call with output=NULL to obtain span count. */
KR_API size_t kr_text_plan_chunks(
    const uint8_t* text,
    size_t length,
    size_t minimum_bytes,
    size_t target_bytes,
    size_t maximum_bytes,
    size_t overlap_bytes,
    kr_text_chunk* output,
    size_t capacity);
KR_API uint32_t kr_format_probe_signature(const uint8_t* bytes, size_t length);
KR_API void kr_numeric_gradient_f32(
    const float* x, const float* errors, size_t rows, size_t features, float* gradient);
KR_API float kr_kernel_dot_f32(const float* left, const float* right, size_t values);
KR_API void* kr_model_create(const kr_model_config* config);
KR_API void kr_model_destroy(void* handle);
KR_API int kr_model_train_step(
    void* handle, const float* x, const float* y, size_t rows, float* loss);
KR_API int kr_model_train_random_step(
    void* handle, const float* x, const float* y, size_t rows, size_t batch_size, float* loss);
KR_API int kr_model_train_random_steps(
    void* handle,
    const float* x,
    const float* y,
    size_t rows,
    size_t batch_size,
    size_t steps,
    float* loss);
KR_API int kr_model_train_steps(
    void* handle,
    const float* x,
    const float* y,
    size_t rows,
    size_t steps,
    float* loss);
KR_API int kr_model_predict(
    const void* handle, const float* x, size_t rows, float* output, size_t output_values);
KR_API size_t kr_model_weight_count(const void* handle);
KR_API size_t kr_model_bias_count(const void* handle);
KR_API int kr_model_export(
    const void* handle, float* weights, size_t weight_count, float* bias, size_t bias_count);
KR_API int kr_model_import(
    void* handle, const float* weights, size_t weight_count, const float* bias, size_t bias_count);
/* Bit mask of engines used by this model since it was created. */
KR_API uint32_t kr_model_execution_mask(const void* handle);

KR_API void* kr_csv_load_numeric(const char* path_utf8, const char* target_utf8, char delimiter);
KR_API void kr_csv_destroy(void* handle);
KR_API size_t kr_csv_rows(const void* handle);
KR_API size_t kr_csv_features(const void* handle);
KR_API const char* kr_csv_target_name(const void* handle);
KR_API const char* kr_csv_feature_name(const void* handle, size_t index);
KR_API float kr_csv_feature_mean(const void* handle, size_t index);
KR_API float kr_csv_feature_std(const void* handle, size_t index);
KR_API int kr_csv_copy(
    const void* handle, float* x, size_t x_values, float* y, size_t y_values);

KR_API void* kr_csv_stream_open(
    const char* path_utf8,
    char delimiter,
    const uint32_t* feature_columns,
    size_t feature_count,
    uint32_t target_column,
    const float* means,
    const float* stds,
    uint32_t task,
    const float* classes,
    size_t class_count,
    uint32_t selected_split,
    uint64_t selected_records);
KR_API void kr_csv_stream_destroy(void* handle);
KR_API int kr_csv_stream_next_batch(
    void* handle, size_t batch_size, float* x, size_t x_values, float* y, size_t y_values);
KR_API uint64_t kr_csv_stream_rows_consumed(const void* handle);
KR_API int kr_csv_stream_restore(void* handle, uint64_t rows_consumed);

KR_API void* kr_csv_scan_numeric(const char* path_utf8, const char* target_utf8, char delimiter);
KR_API void kr_csv_scan_destroy(void* handle);
KR_API uint64_t kr_csv_scan_rows(const void* handle);
KR_API uint64_t kr_csv_scan_split_rows(const void* handle, uint32_t split);
KR_API size_t kr_csv_scan_features(const void* handle);
KR_API size_t kr_csv_scan_columns(const void* handle);
KR_API const char* kr_csv_scan_column_name(const void* handle, size_t index);
KR_API uint32_t kr_csv_scan_target_column(const void* handle);
KR_API const char* kr_csv_scan_target_name(const void* handle);
KR_API const char* kr_csv_scan_feature_name(const void* handle, size_t index);
KR_API float kr_csv_scan_feature_mean(const void* handle, size_t index);
KR_API float kr_csv_scan_feature_std(const void* handle, size_t index);
KR_API size_t kr_csv_scan_target_values(const void* handle);
KR_API const char* kr_csv_scan_target_value(const void* handle, size_t index);

#ifdef __cplusplus
}
#endif
