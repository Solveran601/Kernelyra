#include "execution_guard.h"

#include "checked_arithmetic.h"

int kr_c_core_matrix_elements(size_t rows, size_t columns, size_t* output_elements) {
  if (rows == 0U || columns == 0U) return 0;
  return kr_c_checked_multiply(rows, columns, output_elements);
}
