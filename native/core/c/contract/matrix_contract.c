#include "execution_guard.h"

#include "checked_arithmetic.h"

int kr_c_core_matrix_elements(size_t rows, size_t columns, size_t* output_elements) {
  if (rows == 0U || columns == 0U) return 0;
  return kr_c_checked_multiply(rows, columns, output_elements);
}

int kr_c_core_model_contract_make(
    size_t features, size_t classes, kr_c_model_contract* output) {
  size_t weight_elements = 0U;
  size_t weight_bytes = 0U;
  if (output == NULL || !kr_c_core_matrix_elements(features, classes, &weight_elements) ||
      !kr_c_checked_multiply(weight_elements, sizeof(float), &weight_bytes)) {
    return 0;
  }
  output->weight_elements = weight_elements;
  output->weight_bytes = weight_bytes;
  return 1;
}
