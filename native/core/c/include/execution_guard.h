#ifndef KERNELYRA_C_EXECUTION_GUARD_H
#define KERNELYRA_C_EXECUTION_GUARD_H

#include "kernelyra_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * C owns the smallest trust boundary in the native pipeline. These values
 * are verified before C++, Rust, Zig, or Fortran reserve memory or calculate
 * pointer offsets. The implementation deliberately uses no heap allocation.
 */
typedef struct kr_c_batch_contract {
  size_t dataset_elements;
  size_t batch_elements;
  size_t dataset_bytes;
  size_t batch_bytes;
  size_t working_bytes;
} kr_c_batch_contract;

/* Exact, overflow-checked storage required by one dense model weight table. */
typedef struct kr_c_model_contract {
  size_t weight_elements;
  size_t weight_bytes;
} kr_c_model_contract;

/* Return 1 only when rows * columns is representable as size_t. */
KR_API int kr_c_core_matrix_elements(size_t rows, size_t columns, size_t* output_elements);

/* Return 1 only when a features-by-classes float32 weight table is representable. */
KR_API int kr_c_core_model_contract_make(
    size_t features, size_t classes, kr_c_model_contract* output);

/*
 * Validate a random-training request and publish exact float32 storage
 * requirements. batch_rows must be in 1..dataset_rows. working_bytes is the
 * checked sum of the source table and the temporary feature batch.
 */
KR_API int kr_c_core_batch_contract_make(
    size_t dataset_rows, size_t features, size_t batch_rows, kr_c_batch_contract* output);

#ifdef __cplusplus
}
#endif

#endif
