/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
#define CUTE_ARCH_COPY_XE_ENABLED
#endif

#if defined(CUTE_ARCH_COPY_XE_ENABLED) && ((defined(__INTEL_LLVM_COMPILER) && (__INTEL_LLVM_COMPILER < 20250200)) || defined(CUTLASS_SYCL_BUILTIN_ENABLE))
#include <cute/arch/copy_xe_builtin.hpp>
#elif defined(CUTE_ARCH_COPY_XE_ENABLED)
#include <cute/arch/copy_xe_spirv.hpp>
#endif

#include <cute/arch/copy_xe_U4.hpp>
#include <cute/arch/copy_xe_U8.hpp>
#include <cute/arch/copy_xe_U16.hpp>
#include <cute/arch/copy_xe_U32.hpp>
#include <cute/arch/copy_xe_U64.hpp>

#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL __attribute__((convergent)) void __spirv_ControlBarrierWaitINTEL(int execution_scope, int memory_scope, int memory_semantics);
SYCL_EXTERNAL __attribute__((convergent)) void __spirv_ControlBarrierArriveINTEL(int execution_scope, int memory_scope, int memory_semantics);
#endif

namespace cute
{

// scope = 3 is for subgroup, scop = 2 is for workgroup
CUTE_HOST_DEVICE void barrier_arrive(int scope, int memory_scope = 0, int emory_semantics = 0) { 
#ifdef __SYCL_DEVICE_ONLY__
  __spirv_ControlBarrierArriveINTEL(scope, memory_scope, emory_semantics); 
#endif
}
CUTE_HOST_DEVICE void barrier_wait(int scope, int memory_scope = 0, int emory_semantics = 0) { 
#ifdef __SYCL_DEVICE_ONLY__
  __spirv_ControlBarrierWaitINTEL(scope, memory_scope, emory_semantics); 
#endif
}

template<class S, class D = S>
struct XE_ATOMIC {
  using SRegisters = S[1];
  using DRegisters = D[1];

  CUTE_STATIC_ASSERT(is_same_v<S, float> || is_same_v<S, double> || is_same_v<S, int>);

  template<class S_, class D_>
  CUTE_HOST_DEVICE static void
  copy(S_ const& src, D_ & dst) {
    #if defined(SYCL_INTEL_TARGET)
      auto v = sycl::atomic_ref<D_, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>(*&dst);
      v += static_cast<D_>(*&src);
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-Xe hardware");
    #endif
  }
};

template <class S, class D = S>
struct XE_1D_LDSM {
  using SRegisters = S[1];
  using DRegisters = D[1];

  CUTE_STATIC_ASSERT(sizeof(D) % sizeof(S) == 0,
    "dst failed to vectorize into registers");
  static constexpr size_t N = sizeof(D) / sizeof(S);
  CUTE_STATIC_ASSERT(N == 1 || N == 2 || N == 4 || N == 8,
    "register vector only supports 1, 2, 4, 8");

  template<class S_, class D_>
  CUTE_HOST_DEVICE static void
  copy(const S_ &src, D_ &dst) {
    #if defined(SYCL_INTEL_TARGET)
      CUTE_STATIC_ASSERT(sizeof(S_) == sizeof(S));
      auto props = sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::contiguous_memory,
          sycl::ext::oneapi::experimental::alignment<sizeof(D)>};
      auto sg =
          sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_sub_group();
      sycl::ext::oneapi::experimental::group_load(
          sg, &src, *reinterpret_cast<sycl::vec<S_, N> *>(&dst), props);
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-Xe hardware");
    #endif 
  }
};

template <class S, class D = S>
struct XE_1D_LOAD_GLOBAL {
  using SRegisters = S[1];
  using DRegisters = D[1];

  CUTE_STATIC_ASSERT(sizeof(D) % sizeof(S) == 0,
    "dst failed to vectorize into registers");
  static constexpr size_t N = sizeof(D) / sizeof(S);
  CUTE_STATIC_ASSERT(N == 1 || N == 2 || N == 4 || N == 8,
    "register vector only supports 1, 2, 4, 8");

  template<class S_, class D_>
  CUTE_HOST_DEVICE static void
  copy(const S_ &src, D_ &dst) {
    #if defined(SYCL_INTEL_TARGET)
      CUTE_STATIC_ASSERT(sizeof(S_) == sizeof(S));
      CUTE_STATIC_ASSERT(sizeof(D_) == sizeof(D));
      auto sg = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_sub_group();
      auto props = sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::contiguous_memory,
          sycl::ext::oneapi::experimental::alignment<sizeof(D)>};
      sycl::ext::oneapi::experimental::group_load(
          sg, &src, *reinterpret_cast<sycl::vec<S_, N> *>(&dst), props);
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-Xe hardware");
    #endif 
  }
};

template<class S, class D = S>
struct XE_1D_STSM {
  using SRegisters = S[1];
  using DRegisters = D[1];

  CUTE_STATIC_ASSERT(sizeof(S) % sizeof(D) == 0,
      "src failed to vectorize into registers");
  static constexpr size_t N = sizeof(S) / sizeof(D);
  CUTE_STATIC_ASSERT(N == 1 || N == 2 || N == 4 || N == 8,
      "register vector only supports 1, 2, 4, 8");

  template<class S_, class D_>
  CUTE_HOST_DEVICE static void
  copy(S_ const& src, D_ & dst) {
    #if defined(SYCL_INTEL_TARGET)
      auto sg = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_sub_group();
      auto props = sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::contiguous_memory,
          sycl::ext::oneapi::experimental::alignment<sizeof(S)>};
      sycl::ext::oneapi::experimental::group_store(
          sg, *reinterpret_cast<sycl::vec<D_, N> const *>(&src), &dst, props);
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-Xe hardware");
    #endif
  }
};

template<class S, class D = S>
struct XE_1D_STORE_GLOBAL {
  using SRegisters = S[1];
  using DRegisters = D[1];

  CUTE_STATIC_ASSERT(sizeof(S) % sizeof(D) == 0,
      "src failed to vectorize into registers");
  static constexpr size_t N = sizeof(S) / sizeof(D);
  CUTE_STATIC_ASSERT(N == 1 || N == 2 || N == 4 || N == 8,
      "register vector only supports 1, 2, 4, 8");

  template<class S_, class D_>
  CUTE_HOST_DEVICE static void
  copy(S_ const& src, D_ &dst) {
    #if defined(SYCL_INTEL_TARGET)
      auto sg = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_sub_group();
      auto props = sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::contiguous_memory,
          sycl::ext::oneapi::experimental::alignment<sizeof(S)>};
      sycl::ext::oneapi::experimental::group_store(
          sg, *reinterpret_cast<sycl::vec<D_, N> const *>(&src), &dst, props);
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-Xe hardware");
    #endif
  }
};

template <int BlockHeight, int BlockWidth>
struct XE_2D_Universal_LD_N {
  using BlockShape = Shape<Int<BlockHeight>, Int<BlockWidth>>;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *base_offset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_COPY_XE_ENABLED)
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    const int sg_id = sg.get_local_id()[0];

    CUTE_UNROLL
    for (int y = 0; y < BlockHeight; ++y) {
      int global_height_idx = coord.y + y;
      const char* current_height_ptr = static_cast<const char*>(base_offset) + (global_height_idx * pitch);;

      CUTE_UNROLL
      for (int x = 0; x < BlockWidth / 16; ++x) {
        int source_width_idx = sg_id + (x * 16);
        int global_width_idx = coord.x + source_width_idx;

        T value;
        if (global_height_idx < height  && global_width_idx * sizeof(T) < width) {
          const T* source_element_ptr = reinterpret_cast<const T*>(
              current_height_ptr + (global_width_idx * sizeof(T))
          );
          value = *source_element_ptr;
        } else {
          value = static_cast<T>(0);
        }
        dst[(x * BlockHeight) + y] = value;
      }
    }
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-Xe hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
    }
  };
};

template <int BlockHeight, int BlockWidth>
struct XE_2D_Universal_LD_V : XE_2D_Universal_LD_N<BlockHeight, BlockWidth> {
  using BlockShape = Shape<Int<BlockWidth>, Int<BlockHeight>>;
};

template <int BlockHeight, int BlockWidth>
struct XE_2D_Universal_LD_T {
  using BlockShape = Shape<Int<BlockWidth>, Int<BlockHeight>>;
  static constexpr bool is_transpose = true;

  template <class T>
  CUTE_HOST_DEVICE static void copy(const void *base_offset, int width,
                                    int height, int pitch, intel::coord_t coord,
                                    T *dst) {
#if defined(CUTE_ARCH_COPY_XE_ENABLED)
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    const int sg_id = sg.get_local_id()[0];

    CUTE_UNROLL
    for (int y = 0; y < BlockHeight / 16; ++y) {
      int global_height_idx = coord.y + sg_id + (y * 16);

      const char* current_height_ptr =
          static_cast<const char*>(base_offset) + global_height_idx * pitch ;

      CUTE_UNROLL
      for (int x = 0; x < BlockWidth; ++x) {
        int global_width_idx = coord.x + x;

        T value;
        if (global_width_idx * sizeof(T) < width && global_height_idx < height) {
          const T* source_element_ptr = reinterpret_cast<const T*>(
              current_height_ptr + (global_width_idx * sizeof(T))
          );
          value = *source_element_ptr;
        } else {
          value = static_cast<T>(0);
        }

        dst[(y * BlockWidth) + x] = value;
      }
    }
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use block loads on non-Xe hardware");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(const void *baseoffset, int width,
                                      int height, int pitch,
                                      intel::coord_t coord) {
    }
  };
};

} // end namespace cute
