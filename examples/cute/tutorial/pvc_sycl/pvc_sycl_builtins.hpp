#pragma once

#include "syclcompat/syclcompat.hpp"
#include <cute/util/sycl_vec.hpp>

#define global
#define __global

#define as_long (long)

#define SGS_PER_WG_Y (WG_SIZE_Y / SG_SIZE_Y)
#define SGS_PER_WG_X (WG_SIZE_X / SG_SIZE_X)

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

// Note for 2D block reads:
//  - the tile width and height is encoded into the function name.
//  - base_address is the byte address.  Must be 64B aligned.
//  - width is the width of the entire matrix, in bytes.  Must be >= 64B.  Must
//  be 4B aligned.
//  - height is the height of the entire matrix, or equivalently the number of
//  rows.
//  - pitch is the number of bytes between rows of the entire matrix.  Must be
//  >= 64B.  Must be a multiple of 8 bytes.
//  - coord is the number of elements (x coord) and row (y coord) to read from.
//  X coord must be multiple 4 for for 1B data and 2 for 2B data.

// Built-in functions are:

// #ifdef cl_intel_subgroup_extended_block_read
// ushort2  intel_subgroup_block_read_u8_m1k32v2(__global void *base_address,
// int width, int height, int pitch, int2 coord); ushort4
// intel_subgroup_block_read_u8_m2k32v2(__global void *base_address, int width,
// int height, int pitch, int2 coord); ushort8
// intel_subgroup_block_read_u8_m4k32v2(__global void *base_address, int width,
// int height, int pitch, int2 coord); ushort16
// intel_subgroup_block_read_u8_m8k32v2(__global void *base_address, int width,
// int height, int pitch, int2 coord); ushort2
// intel_subgroup_block_read_u16_m1k16v2(__global void *base_address, int width,
// int height, int pitch, int2 coord); ushort4
// intel_subgroup_block_read_u16_m2k16v2(__global void *base_address, int width,
// int height, int pitch, int2 coord); ushort8
// intel_subgroup_block_read_u16_m4k16v2(__global void *base_address, int width,
// int height, int pitch, int2 coord); ushort16
// intel_subgroup_block_read_u16_m8k16v2(__global void *base_address, int width,
// int height, int pitch, int2 coord); uint8
// intel_subgroup_block_read_transform_u8_k32(__global void *base_address, int
// width, int height, int pitch, int2 coord); uint8
// intel_subgroup_block_read_transform_u16_k16(__global void *base_address, int
// width, int height, int pitch, int2 coord); uint8
// intel_subgroup_block_read_transpose_u32_k8(__global void *base_address, int
// width, int height, int pitch, int2 coord); ulong4
// intel_subgroup_block_read_transpose_u64_k4(__global void *base_address, int
// width, int height, int pitch, int2 coord); #endif
// //defined(cl_intel_subgroup_extended_block_read)

// For intrinsics, the pattern is:
//  - prefix: __builtin_IB_subgroup_block_read_flat or
//  __builtin_IB_subgroup_block_write_flat
//  - operation (optional): _transpose or _transform
//  - for no transpose or transform:
//      - type / elements size: _u8 or _u16 or _u32 or _u64
//      - number of tile rows: _m32 or _m16 or _m8 or _m4 or _m2 or _m1
//      - tile width: _k64 or _k32 or _k16 or _k8
//      - number of tiles: _v2 or _v1
//  - for transpose:
//      - type / element size: _u64 or _u32
//      - number of tile rows: subgroup size (16)
//      - tile width: _k4 (for _u64) or _k8 (for _u32)
//      - number of tiles: 1
//  - for transform:
//      - type / element size: _u16 or _u8
//      - number of tile rows: _k32 (for _u8) or _k16 (for _u16)
//      - tile width: subgroup size (16)
//      - number of tiles: 1

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

typedef ushort __attribute__((ext_vector_type(32))) ushort32;
typedef uint __attribute__((ext_vector_type(32))) uint32;

typedef ushort __attribute__((ext_vector_type(64))) ushort64;
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
