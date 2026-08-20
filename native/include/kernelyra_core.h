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

enum { KR_ABI_VERSION = 5 };
enum { KR_TASK_BINARY = 0, KR_TASK_MULTICLASS = 1, KR_TASK_REGRESSION = 2 };
enum { KR_SPLIT_TRAIN = 0, KR_SPLIT_VALIDATION = 1, KR_SPLIT_TEST = 2 };
enum {
  KR_COMPONENT_ZIG_MEMORY = 1U,
  KR_COMPONENT_FORTRAN_NUMERIC = 2U,
  KR_COMPONENT_RUST_POLICY = 4U,
  KR_COMPONENT_ALL = 7U
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

KR_API const char* kr_core_version(void);
KR_API const char* kr_core_features(void);
KR_API const char* kr_core_components(void);
KR_API uint32_t kr_core_component_mask(void);
KR_API uint32_t kr_core_enabled_component_mask(void);
KR_API uint32_t kr_core_set_component_mask(uint32_t mask);
KR_API const char* kr_last_error(void);
KR_API void* kr_memory_alloc_aligned(size_t bytes, size_t alignment);
KR_API void kr_memory_free_aligned(void* pointer);
KR_API void kr_memory_normalize_f32(
    float* data, size_t rows, size_t features, const float* means, const float* stds);
KR_API void kr_memory_copy_f32(float* destination, const float* source, size_t values);
KR_API void kr_memory_zero_f32(float* destination, size_t values);
KR_API uint32_t kr_values_all_finite_f32(const float* values, size_t count);
KR_API float kr_values_l2_norm_f32(const float* values, size_t count);
KR_API void kr_values_clip_f32(float* values, size_t count, float limit);
KR_API uint64_t kr_rust_mix_u64(uint64_t value);
KR_API uint32_t kr_rust_split_for_key(
    uint64_t group_key, uint32_t validation_percent, uint32_t test_percent);
KR_API size_t kr_rust_next_chunk_size(
    size_t remaining_records,
    size_t target_records,
    size_t minimum_records,
    size_t maximum_records,
    uint64_t sequence,
    uint64_t seed);
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
KR_API int kr_model_predict(
    const void* handle, const float* x, size_t rows, float* output, size_t output_values);
KR_API size_t kr_model_weight_count(const void* handle);
KR_API size_t kr_model_bias_count(const void* handle);
KR_API int kr_model_export(
    const void* handle, float* weights, size_t weight_count, float* bias, size_t bias_count);
KR_API int kr_model_import(
    void* handle, const float* weights, size_t weight_count, const float* bias, size_t bias_count);

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
