#ifndef KERNELYRA_C_CHECKED_ARITHMETIC_H
#define KERNELYRA_C_CHECKED_ARITHMETIC_H

#include <stddef.h>

/* Internal overflow-safe primitives shared by C contract modules. */
int kr_c_checked_add(size_t left, size_t right, size_t* output);
int kr_c_checked_multiply(size_t left, size_t right, size_t* output);

#endif
