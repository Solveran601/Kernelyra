#include "execution_guard.h"

#include <stdint.h>

static int kr_c_checked_multiply(size_t left, size_t right, size_t* output) {
  if (output == NULL) return 0;
  if (left != 0U && right > SIZE_MAX / left) return 0;
  *output = left * right;
  return 1;
}

int kr_c_core_matrix_elements(size_t rows, size_t columns, size_t* output_elements) {
  if (rows == 0U || columns == 0U) return 0;
  return kr_c_checked_multiply(rows, columns, output_elements);
}

int kr_c_core_batch_contract_make(
    size_t dataset_rows, size_t features, size_t batch_rows, kr_c_batch_contract* output) {
  size_t dataset_elements = 0U;
  size_t batch_elements = 0U;
  size_t dataset_bytes = 0U;
  size_t batch_bytes = 0U;
  if (output == NULL || batch_rows == 0U || batch_rows > dataset_rows ||
      !kr_c_core_matrix_elements(dataset_rows, features, &dataset_elements) ||
      !kr_c_core_matrix_elements(batch_rows, features, &batch_elements) ||
      !kr_c_checked_multiply(dataset_elements, sizeof(float), &dataset_bytes) ||
      !kr_c_checked_multiply(batch_elements, sizeof(float), &batch_bytes)) {
    return 0;
  }
  output->dataset_elements = dataset_elements;
  output->batch_elements = batch_elements;
  output->dataset_bytes = dataset_bytes;
  output->batch_bytes = batch_bytes;
  return 1;
}
