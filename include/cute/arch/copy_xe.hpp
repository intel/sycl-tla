#pragma once

#include <array>
#include <cute/arch/copy.hpp>
#include <cute/config.hpp>
#include <cute/util/sycl_vec.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_DEVICE_BUILTIN(x) SYCL_EXTERNAL extern "C" x
#else
#define SYCL_DEVICE_BUILTIN(x)                                                 \
  inline x { assert(false); }
#endif

SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord, uint8 data));
SYCL_DEVICE_BUILTIN(ushort8 __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));
SYCL_DEVICE_BUILTIN(uint8 __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));

/// Load A
SYCL_DEVICE_BUILTIN(ushort64 __builtin_IB_subgroup_block_read_flat_u16_m32k16v2(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));
SYCL_DEVICE_BUILTIN(ushort32 __builtin_IB_subgroup_block_read_flat_u16_m16k16v2(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));
SYCL_DEVICE_BUILTIN(ushort16 intel_subgroup_block_read_u16_m8k16v2(
    __global void *baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));
SYCL_DEVICE_BUILTIN(ushort32 __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));

/// Load B
SYCL_DEVICE_BUILTIN(uint16 __builtin_IB_subgroup_block_read_flat_u32_m16k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2_ coord));

#undef SYCL_DEVICE_BUILTIN

struct XE_2D_LOAD // m8k16
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    if constexpr (sizeof(T) == sizeof(ushort)) {
      *(ushort8 *)dst = __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord);
    } else if constexpr (sizeof(T) == sizeof(uint)) {
      *(uint8 *)dst = __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord);
    } else {
      static_assert(false);
    }
  }
};

/// 4X2 Block m8k16
struct XE_2D_U16X4X2_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    if constexpr (sizeof(T) == 2) {
      *(ushort64 *)dst = __builtin_IB_subgroup_block_read_flat_u16_m32k16v2(
          long(baseoffset), width - 1, height - 1, pitch - 1, coord);

      // ((ushort8_t*)dst)[0] = sycl::bit_cast<ushort8_t>(tmp.lo.lo.lo);
      // ((ushort8_t*)dst)[1] = sycl::bit_cast<ushort8_t>(tmp.lo.lo.hi);
      // ((ushort8_t*)dst)[2] = sycl::bit_cast<ushort8_t>(tmp.lo.hi.lo);
      // ((ushort8_t*)dst)[3] = sycl::bit_cast<ushort8_t>(tmp.lo.hi.hi);
      // ((ushort8_t*)dst)[4] = sycl::bit_cast<ushort8_t>(tmp.hi.lo.lo);
      // ((ushort8_t*)dst)[5] = sycl::bit_cast<ushort8_t>(tmp.hi.lo.hi);
      // ((ushort8_t*)dst)[6] = sycl::bit_cast<ushort8_t>(tmp.hi.hi.lo);
      // ((ushort8_t*)dst)[7] = sycl::bit_cast<ushort8_t>(tmp.hi.hi.hi);
    } else {
      static_assert(false);
    }
  }
};

/// 2X2 Block m8k16
struct XE_2D_U16X2X2_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    if constexpr (sizeof(T) == 2) {
      ushort32 tmp = __builtin_IB_subgroup_block_read_flat_u16_m16k16v2(
          long(baseoffset), width - 1, height - 1, pitch - 1, coord);
      ((ushort8_t *)dst)[0] = (ushort8_t)(tmp.lo.lo);
      ((ushort8_t *)dst)[1] = (ushort8_t)(tmp.lo.hi);
      ((ushort8_t *)dst)[2] = (ushort8_t)(tmp.hi.lo);
      ((ushort8_t *)dst)[3] = (ushort8_t)(tmp.hi.hi);
    } else {
      static_assert(false);
    }
  }
};

/// 1X2 Block m8k16
struct XE_2D_U16X1X2_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    if constexpr (sizeof(T) == 2) {
      ushort16 tmp = (intel_subgroup_block_read_u16_m8k16v2(
          (__global void *)baseoffset, width, height, pitch, coord));
      *(ushort16 *)dst = *reinterpret_cast<ushort16 *>(&tmp);
    } else {
      static_assert(false);
    }
  }
};

/// 4X1 Block m8k16
struct XE_2D_U16X4X1_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    if constexpr (sizeof(T) == 2) {
      ushort32 tmp = __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(
          long(baseoffset), width - 1, height - 1, pitch - 1, coord);
      ((ushort8_t *)dst)[0] = (ushort8_t)(tmp.lo.lo);
      ((ushort8_t *)dst)[1] = (ushort8_t)(tmp.lo.hi);
      ((ushort8_t *)dst)[2] = (ushort8_t)(tmp.hi.lo);
      ((ushort8_t *)dst)[3] = (ushort8_t)(tmp.hi.hi);

    } else {
      static_assert(false);
    }
  }
};

/// 2X1 BLock U32 k8n16
struct XE_2D_U32X2X1_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    if constexpr (sizeof(T) == 4) {
      uint16 tmp = __builtin_IB_subgroup_block_read_flat_u32_m16k16v1(
          long(baseoffset), width - 1, height - 1, pitch - 1, coord);
      *(uint16 *)dst = *reinterpret_cast<uint16 *>(&tmp);
    } else {
      static_assert(false);
    }
  }
};

/// 2X1 Block U16 k16n16
struct XE_2D_U16X2X1_N {
  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                    int height, int pitch, int2_ coord,
                                    T *dst) {
    if constexpr (true) {
      uint16 tmp = __builtin_IB_subgroup_block_read_flat_u32_m16k16v1(
          long(baseoffset), width - 1, height - 1, pitch - 1, coord);
      *(uint16 *)dst = *reinterpret_cast<uint16 *>(&tmp);
    } else {
      static_assert(false);
    }
  }
};

struct XE_2D_SAVE // m8k16
{
  template <class T>
  CUTE_HOST_DEVICE static void copy(void *baseoffset, int width, int height,
                                    int pitch, int2_ coord, const T *src) {
    if constexpr (sizeof(T) == sizeof(uint)) {
      __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(
          (long)baseoffset, width - 1, height - 1, pitch - 1, coord,
          *(uint8 *)src);
    } else {
      static_assert(false);
    }
  }
};
