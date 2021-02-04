#ifndef NN_CODE_HELPER_H
#define NN_CODE_HELPER_H

#ifndef VERSION
#define VERSION "0.0.1"
#endif

// Global Includes
#include <cstdint>      // needed for uint8_t and so on
#include <cassert>
#include <vector>       // for __level_translation_array
#include <array>

#if __AVX2__
#include <immintrin.h>
#include <emmintrin.h>
#endif

// optimisation flags/commands
#define STRINGIFY(a) #a

// performance helpers
#if defined(USE_BRANCH_PREDICTION) && (!defined(DEBUG))
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#else
#define likely(x)       x
#define unlikely(x)     x
#endif

#if defined(USE_PREFETCH)  && (!defined(DEBUG))
/*
 * The value of addr is the address of the memory to prefetch. There are two optional arguments, rw and locality.
 * The value of rw is a compile-time constant one or zero; one means that the prefetch is preparing for a write to the
 * memory address and zero the default, means that the prefetch is preparing for a read. The value locality must be a
 * compile-time constant integer between zero and three. A value of zero means that the data has no temporal locality,
 * so it need not be left in the cache after the access. A value of three means that the data has a high degree of
 * temporal locality and should be left in all levels of cache possible. Values of one and two mean, respectively,
 * a low or moderate degree of temporal locality. The default is three.
 */
#define prefetch(m, x, y) __builtin_prefetch(m, x, y)
#else
#define prefetch(m, x, y)
#endif

#ifdef USE_LOOP_UNROLL
// Some loop unrollment optimisation
#define LOOP_UNROLL()                      \
    _Pragma(STRINGIFY(clang loop unroll(full)))
#else
#define LOOP_UNROLL()
#endif

#include <iostream>

#ifdef DEBUG
#include <cassert>
#define ASSERT(x) assert(x)
#else
#define ASSERT(x)
#endif

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#endif //NN_CODE_HELPER_H
