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

#include <sycl/sycl.hpp>
#include <syclcompat>

namespace syclcompat::ext {

using namespace sycl;
using namespace syclcompat;

namespace detail {

template <typename F, typename... Args>
inline constexpr bool is_invocable {
  return std::is_invocable_v<decltype(F), Args...>;
}

template <typename R, typename... Types>
constexpr size_t getArgumentCount(R (*f)(Types...)) {
  return sizeof...(Types);
}

template <int SubgroupSize>
struct SubgroupSizeStruct {
  constexpr static int _SubgroupSize = SubgroupSize;
};

// Introduced in https://github.com/intel/llvm/pull/11192/

template <int MaxThreadsPerBlock, int MinBlocksPerSM>
struct NvidiaLaunchBoundsStruct {
  constexpr static int _MaxThreadsPerBlock = MaxThreadsPerBlock;
  constexpr static int _MinBlocksPerSM = MinBlocksPerSM;
};

template <int MaxThreadsPerBlock, int MinBlocksPerSM, int MaxWgClusterSize>
struct NvidiaLaunchBoundsWithClusterStruct {
  constexpr static int _MaxThreadsPerBlock = MaxThreadsPerBlock;
  constexpr static int _MinBlocksPerSM = MinBlocksPerSM;
  constexpr static int _MaxWgClusterSize = MaxWgClusterSize;
};

template <class SubgroupSizeStruct, auto F, typename... Args>
sycl::event launch(const nd_range<3>& launch_config, sycl::queue& q, std::size_t local_memory_size,
                   Args... args) {
  static_assert(std::is_same_v<typename std::invoke_result_t<decltype(F), Args...>, void>,
                "SYCL Kernels cannot have non void return type");
  return queue.submit([&](handler& cgh) {
    using SubgroupSize = SubgroupSizeStruct::_SubgroupSize;
    if constexpr (getArgumentCount(F) == sizeof...(Args) + 1) {
      local_accessor<char, 1> local_mem(local_memory_size, cgh);
      cgh.parallel_for(launch_config,
                       [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
                         auto local_mem_ptr = local_mem.get_pointer();
                         [[clang::always_inline]] F(args..., local_mem_ptr);
                       });
    } else {
      cgh.parallel_for(launch_config,
                       [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
                         auto local_mem_ptr = local_mem.get_pointer();
                         [[clang::always_inline]] F(args...);
                       });
    }
  });
}

template <class NvidiaLaunchBoundsStruct, auto F, typename... Args>
sycl::event launch(const nd_range<3>& launch_config, sycl::queue& q, std::size_t local_memory_size,
                   Args... args) {
  static_assert(std::is_same_v<typename std::invoke_result_t<decltype(F), Args...>, void>,
                "SYCL Kernels cannot have non void return type");
  return queue.submit([&](handler& cgh) {
    using MaxWorkGroupSize = NvidiaLaunchBoundsStruct::_MaxThreadsPerBlock;
    using MinBlocksPerSM = NvidiaLaunchBoundsStruct::_MinBlocksPerSM;

    if constexpr (getArgumentCount(F) == sizeof...(Args) + 1) {
      local_accessor<char, 1> local_mem(local_memory_size, cgh);
      cgh.parallel_for(
          launch_config, [=](sycl::nd_item<3> it)
                             __attribute__((intel::max_work_group_size(MaxWorkGroupSize)),
                                           (intel::min_work_groups_per_cu(MinBlocksPerSM))) {
                               auto local_mem_ptr = local_mem.get_pointer();
                               [[clang::always_inline]] F(args..., local_mem_ptr);
                             });
    } else {
      cgh.parallel_for(
          launch_config, [=](sycl::nd_item<3> it)
                             __attribute__((intel::max_work_group_size(MaxWorkGroupSize)),
                                           (intel::min_work_groups_per_cu(MinBlocksPerSM))) {
                               auto local_mem_ptr = local_mem.get_pointer();
                               [[clang::always_inline]] F(args...);
                             });
    }
  });
}

template <class NvidiaLaunchBoundsWithClusterStruct, auto F, typename... Args>
sycl::event launch(const nd_range<3>& launch_config, sycl::queue& q, std::size_t local_memory_size,
                   Args... args) {
  static_assert(std::is_same_v<typename std::invoke_result_t<decltype(F), Args...>, void>,
                "SYCL Kernels cannot have non void return type");
  return queue.submit([&](handler& cgh) {
    using MaxWorkGroupSize = NvidiaLaunchBoundsWithClusterStruct::_MaxThreadsPerBlock;
    using MinBlocksPerSM = NvidiaLaunchBoundsWithClusterStruct::_MinBlocksPerSM;
    using MaxWgClusterSize = NNvidiaLaunchBoundsWithClusterStruct::_MaxWgClusterSize;

    if constexpr (getArgumentCount(F) == sizeof...(Args) + 1) {
      local_accessor<char, 1> local_mem(local_memory_size, cgh);
      cgh.parallel_for(
          launch_config, [=](sycl::nd_item<3> it)
                             __attribute__((intel::max_work_group_size(MaxWorkGroupSize)),
                                           (intel::min_work_groups_per_cu(MinBlocksPerSM)),
                                           (intel::max_work_groups_per_mp(MaxWgClusterSize))) {
                               auto local_mem_ptr = local_mem.get_pointer();
                               [[clang::always_inline]] F(args..., local_mem_ptr);
                             });
    } else {
      cgh.parallel_for(
          launch_config, [=](sycl::nd_item<3> it)
                             __attribute__((intel::max_work_group_size(MaxWorkGroupSize)),
                                           (intel::min_work_groups_per_cu(MinBlocksPerSM)),
                                           (intel::max_work_groups_per_mp(MaxWgClusterSize))) {
                               auto local_mem_ptr = local_mem.get_pointer();
                               [[clang::always_inline]] F(args...);
                             });
    }
  });
}

}  // namespace detail

// Quick Helper Structs
struct IntelSubgroupSizeStruct_16 : detail::SubgroupSizeStruct<16> {};

// API entry points, only providing dim3 types, and using default queue only

template <class SubgroupSizeStruct, auto F, typename... Args>
std::enable_if_t<is_invocable<F, Args..., char*>(), sycl::event> launch(
    const dim3& grid_dim, const dim3& block_dim, std::size_t local_memory_size, Args... args) {
  return detail::launch<SubgroupSizeStruct, F>(sycl::nd_range<3>{grid_dim * block_dim, block_dim},
                                               get_default_queue(), local_memory_size, args...);
}

template <class SubgroupSizeStruct, auto F, typename... Args>
std::enable_if_t<is_invocable<F, Args...>(), sycl::event> launch(const dim3& grid_dim,
                                                                 const dim3& block_dim,
                                                                 Args... args) {
  return detail::launch<SubgroupSizeStruct, F>(sycl::nd_range<3>{grid_dim * block_dim, block_dim},
                                               get_default_queue(), 0, args...);
}

template <class NvidiaLaunchBoundsStruct, auto F, typename... Args>
std::enable_if_t<is_invocable<F, Args..., char*>(), sycl::event> launch(
    const dim3& grid_dim, const dim3& block_dim, std::size_t local_memory_size, Args... args) {
  return detail::launch<NvidiaLaunchBoundsStruct, F>(
      sycl::nd_range<3>{grid_dim * block_dim, block_dim}, get_default_queue(), local_memory_size,
      args...);
}

template <class NvidiaLaunchBoundsStruct, auto F, typename... Args>
std::enable_if_t<is_invocable<F, Args...>(), sycl::event> launch(const dim3& grid_dim,
                                                                 const dim3& block_dim,
                                                                 Args... args) {
  return detail::launch<NvidiaLaunchBoundsStruct, F>(
      sycl::nd_range<3>{grid_dim * block_dim, block_dim}, get_default_queue(), 0, args...);
}

template <class NvidiaLaunchBoundsWithClusterStruct, auto F, typename... Args>
std::enable_if_t<is_invocable<F, Args..., char*>(), sycl::event> launch(
    const dim3& grid_dim, const dim3& block_dim, std::size_t local_memory_size, Args... args) {
  return detail::launch<NvidiaLaunchBoundsWithClusterStruct, F>(
      sycl::nd_range<3>{grid_dim * block_dim, block_dim}, get_default_queue(), local_memory_size,
      args...);
}

template <class NvidiaLaunchBoundsWithClusterStruct, auto F, typename... Args>
std::enable_if_t<is_invocable<F, Args...>(), sycl::event> launch(const dim3& grid_dim,
                                                                 const dim3& block_dim,
                                                                 Args... args) {
  return detail::launch<NvidiaLaunchBoundsWithClusterStruct, F>(
      sycl::nd_range<3>{grid_dim * block_dim, block_dim}, get_default_queue(), 0, args...);
}

}  // namespace syclcompat::ext