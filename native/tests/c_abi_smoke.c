#include "kernelyra_core.h"
#include "kernelyra_policy.h"

#include <stdint.h>
#include <stdio.h>

int main(void) {
  const uint32_t first = kr_rust_split_for_key(UINT64_C(42), 15U, 15U);
  const uint32_t second = kr_rust_split_for_key(UINT64_C(42), 15U, 15U);
  const size_t chunk = kr_rust_next_chunk_size(10000U, 512U, 256U, 768U, 3U, 99U);
  kr_c_chunk_cursor cursor;
  if (kr_c_chunk_cursor_init(&cursor, 10000U, 512U, 256U, 768U, 99U) != KR_C_POLICY_OK ||
      KR_ABI_VERSION != 5 || first != second || first > KR_SPLIT_TEST ||
      kr_c_context_split(UINT64_C(42), 15U, 15U) != (int)first || chunk < 256U || chunk > 768U ||
      kr_c_chunk_cursor_next(&cursor) < 256U) {
    fputs("Kernelyra C ABI smoke test failed\n", stderr);
    return 1;
  }
  printf("kernelyra C ABI %d: split=%u chunk=%zu\n", KR_ABI_VERSION, first, chunk);
  return 0;
}
