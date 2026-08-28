#include "checked_arithmetic.h"

#include <stdint.h>

int kr_c_checked_add(size_t left, size_t right, size_t* output) {
  if (output == NULL || right > SIZE_MAX - left) return 0;
  *output = left + right;
  return 1;
}

int kr_c_checked_multiply(size_t left, size_t right, size_t* output) {
  if (output == NULL || (left != 0U && right > SIZE_MAX / left)) return 0;
  *output = left * right;
  return 1;
}
