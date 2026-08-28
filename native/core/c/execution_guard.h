#ifndef KERNELYRA_C_EXECUTION_GUARD_H
#define KERNELYRA_C_EXECUTION_GUARD_H

#include "kernelyra_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * C owns the smallest trust boundary in the native pipeline.  These helpers
 * validate dimensions before C++, Rust, Zig, or Fortran reserve memory or
 * calculate pointer offsets.  They deliberately use no heap allocation.
 */
typedef struct kr_c_batch_contract {
  size_t dataset_elements;
  size_t batch_elements;
  size_t dataset_bytes;
  size_t batch_bytes;
} kr_c_batch_contract;

/* Return 1 only when rows * columns is representable as size_t. */
KR_API int kr_c_core_matrix_elements(size_t rows, size_t columns, size_t* output_elements);

/*
 * Validate a random-training request and publish exact float32 storage
 * requirements.  batch_rows must be in 1..dataset_rows.
 */
KR_API int kr_c_core_batch_contract_make(
    size_t dataset_rows, size_t features, size_t batch_rows, kr_c_batch_contract* output);

#ifdef __cplusplus
}
#endif

#endif
