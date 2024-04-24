#pragma once

#include "syclcompat/syclcompat.hpp"
#include <cute/util/sycl_vec.hpp>

#define global
#define __global

#define SGS_PER_WG_X 4
#define SGS_PER_WG_Y 8

#define as_long (long)

#define get_sub_group_id()                                                     \
  (sycl::ext::oneapi::experimental::this_nd_item<3>()                          \
       .get_sub_group()                                                        \
       .get_group_id()[0])
#define get_sub_group_local_id()                                               \
  (sycl::ext::oneapi::experimental::this_nd_item<3>()                          \
       .get_sub_group()                                                        \
       .get_local_id()[0])

#define BLOCK_PREFETCH_CACHE_TYPE LSC_LDCC_L1C_L3C

void prefetch(const void *Ptr, size_t Count);

typedef global ushort *global_aligned_ushort_ptr
    __attribute__((align_value(4)));

// M rows x K columns x V tiles (in the M and K dimensions)
void prefetch_a_rowmajor_d16_m8v2_k16v2_sg16(global ushort *A, int rowStart,
                                             int colStart, int stride) {
#if defined(PREFETCH_DEFAULT)
  uint offset = colStart + (rowStart + get_sub_group_local_id()) * stride;
  __builtin_assume((ulong)(A + offset) % 4 == 0);
  prefetch(A + offset, 2);
#endif // defined(PREFETCH_DEFAULT)
}

// K rows x N columns x V tiles (in the N dimension)
void prefetch_b_rowmajor_d16_k16_n16v2_sg16(global ushort *B, int rowStart,
                                            int colStart, int stride) {
#if defined(PREFETCH_DEFAULT)
  uint offset = colStart + (rowStart + get_sub_group_local_id()) * stride;
  __builtin_assume((ulong)(B + offset) % 4 == 0);
  prefetch(B + offset, 2);
#endif // defined(PREFETCH_DEFAULT)
}

// K rows x N columns x V tiles (in the K dimension)
void prefetch_b_vnni_d16_k16v2_n16_sg16(global ushort *B, int rowStart,
                                        int colStart, int stride) {
#if defined(PREFETCH_DEFAULT)
  global uint *B_ui = (global uint *)B;
  uint offset_ui =
      colStart + (rowStart / 2 + get_sub_group_local_id()) * stride;
  __builtin_assume((ulong)(B_ui + offset_ui) % 4 == 0);
  prefetch(B_ui + offset_ui, 1);
#endif // defined(PREFETCH_DEFAULT)
}

enum LSC_LDCC {
  LSC_LDCC_DEFAULT = 0,
  LSC_LDCC_L1UC_L3UC = 1, // Override to L1 uncached and L3 uncached
  LSC_LDCC_L1UC_L3C = 2,  // Override to L1 uncached and L3 cached
  LSC_LDCC_L1C_L3UC = 3,  // Override to L1 cached and L3 uncached
  LSC_LDCC_L1C_L3C = 4,   // Override to L1 cached and L3 cached
  LSC_LDCC_L1S_L3UC = 5,  // Override to L1 streaming load and L3 uncached
  LSC_LDCC_L1S_L3C = 6,   // Override to L1 streaming load and L3 cached
  LSC_LDCC_L1IAR_L3C = 7, // Override to L1 invalidate-after-read, and L3 cached
};

// typedef ushort __attribute__((ext_vector_type(32))) ushort32;
// typedef uint __attribute__((ext_vector_type(32))) uint32;
// typedef ushort __attribute__((ext_vector_type(64))) ushort64;
typedef uint __attribute__((ext_vector_type(16))) uint16;
typedef uint __attribute__((ext_vector_type(8))) uint8;
typedef int __attribute__((ext_vector_type(8))) int8;
typedef ushort __attribute__((ext_vector_type(8))) ushort8;
typedef short __attribute__((ext_vector_type(8))) short8;
typedef float __attribute__((ext_vector_type(8))) float8;

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_DEVICE_BUILTIN(x) SYCL_EXTERNAL extern "C" x
#else
#define SYCL_DEVICE_BUILTIN(x)                                                 \
  inline x { assert(false); }
#endif

// Define block reads, prefetches, and writes.  These are supported by the
// hardware but are not in the headers:

SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord, enum LSC_LDCC cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v2(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord, enum LSC_LDCC cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord, enum LSC_LDCC cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord, enum LSC_LDCC cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v2(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord, enum LSC_LDCC cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v2(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord, enum LSC_LDCC cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u32_m8k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord, enum LSC_LDCC cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u32_m16k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord, enum LSC_LDCC cache_control));
SYCL_DEVICE_BUILTIN(void __builtin_IB_work_group_barrier_arrive(uint flags));
SYCL_DEVICE_BUILTIN(void __builtin_IB_work_group_barrier_wait(uint flags));

void intel_subgroup_block_prefetch_u16_m8k16(const __global void *base_address,
                                             int width, int height, int pitch,
                                             int2_ coord) {
#if defined(PREFETCH_DEFAULT)
  __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v1(
      as_long(base_address), width - 1, height - 1, pitch - 1, coord,
      BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
void intel_subgroup_block_prefetch_u16_m8k16v2(__global void *base_address,
                                               int width, int height, int pitch,
                                               int2_ coord) {
#if defined(PREFETCH_DEFAULT)
  __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v2(
      as_long(base_address), width - 1, height - 1, pitch - 1, coord,
      BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
void intel_subgroup_block_prefetch_u16_m16k16(const __global void *base_address,
                                              int width, int height, int pitch,
                                              int2_ coord) {
#if defined(PREFETCH_DEFAULT)
  __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v1(
      as_long(base_address), width - 1, height - 1, pitch - 1, coord,
      BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
void intel_subgroup_block_prefetch_u16_m32k16(const __global void *base_address,
                                              int width, int height, int pitch,
                                              int2_ coord) {
#if defined(PREFETCH_DEFAULT)
  __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v1(
      as_long(base_address), width - 1, height - 1, pitch - 1, coord,
      BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
void intel_subgroup_block_prefetch_u16_m16k16v2(
    const __global void *base_address, int width, int height, int pitch,
    int2_ coord) {
#if defined(PREFETCH_DEFAULT)
  __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v2(
      as_long(base_address), width - 1, height - 1, pitch - 1, coord,
      BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
void intel_subgroup_block_prefetch_u16_m32k16v2(
    const __global void *base_address, int width, int height, int pitch,
    int2_ coord) {
#if defined(PREFETCH_DEFAULT)
  __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v2(
      as_long(base_address), width - 1, height - 1, pitch - 1, coord,
      BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
void intel_subgroup_block_prefetch_u32_m8k16(const __global void *base_address,
                                             int width, int height, int pitch,
                                             int2_ coord) {
#if defined(PREFETCH_DEFAULT)
  __builtin_IB_subgroup_block_read_prefetch_u32_m8k16v1(
      as_long(base_address), width - 1, height - 1, pitch - 1, coord,
      BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
void intel_subgroup_block_prefetch_u32_m16k16(const __global void *base_address,
                                              int width, int height, int pitch,
                                              int2_ coord) {
#if defined(PREFETCH_DEFAULT)
  __builtin_IB_subgroup_block_read_prefetch_u32_m16k16v1(
      as_long(base_address), width - 1, height - 1, pitch - 1, coord,
      BLOCK_PREFETCH_CACHE_TYPE);
#endif // defined(PREFETCH_DEFAULT)
}
