#pragma once

// fwd declare OCL function and OCL types
#include <sycl.hpp> //for sycl::vec

#ifdef __SYCL_DEVICE_ONLY__
template <class T, int N> using vector_t = typename sycl::vec<T, N>::vector_t;
#else
template <class T, int N> using vector_t = sycl::vec<T, N>;
#endif

// using float8 = vector_t<float, 8>;
// using short8 = vector_t<short, 8>;
// using ushort8 = vector_t<ushort, 8>;
using int2_ = vector_t<int, 2>; // conflicts with vector_types
// using int8 = vector_t<int, 8>;
// using uint8 = vector_t<uint, 8>;
// using ushort16 = vector_t<ushort, 16>;
// using uint16 = vector_t<uint, 16>;

typedef ushort __attribute__((ext_vector_type(8))) ushort8_t;
typedef ushort __attribute__((ext_vector_type(16))) ushort16;
typedef ushort __attribute__((ext_vector_type(32))) ushort32;
typedef ushort __attribute__((ext_vector_type(64))) ushort64;
typedef uint __attribute__((ext_vector_type(32))) uint32;
