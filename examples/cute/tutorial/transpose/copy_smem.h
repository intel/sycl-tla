#pragma once
/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <cute/util/compat.hpp>
#include <sycl/sycl.hpp>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "util.h"

#include "cutlass/detail/layout.hpp"

// Shared Storage for aligned addresses
template <class Element, class SmemLayout> struct SharedStorageCopy {
  cute::array_aligned<Element, cute::cosize_v<SmemLayout>> smem;
};

template <class TensorS, class TensorD, class ThreadLayout, class SmemLayout,
          int smem_size>
void copySmemKernel(TensorS const S, TensorD const D, ThreadLayout,
                    SmemLayout) {
  using namespace cute;
  using Element = typename TensorS::value_type;

  // Use Shared Storage structure to allocate aligned SMEM addresses.
  using SharedStorage = SharedStorageCopy<Element, SmemLayout>;
  auto smem = compat::local_mem<Element[smem_size]>();
  SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem);

  Tensor gS = S(make_coord(_, _), compat::work_group_id::x(),
                compat::work_group_id::y()); // (bM, bN)
  Tensor gD = D(make_coord(_, _), compat::work_group_id::x(),
                compat::work_group_id::y()); // (bM, bN)

  Tensor sS = make_tensor(make_smem_ptr(shared_storage.smem.data()),
                          SmemLayout{}); // (bM, bN)

  auto tiled_copy_load = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
      ThreadLayout{});

  auto tiled_copy_store = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
      ThreadLayout{});
  //
  // Construct a Tensor corresponding to each thread's slice.
  auto thr_copy_load = tiled_copy_load.get_thread_slice(compat::local_id::x());
  auto thr_copy_store =
      tiled_copy_store.get_thread_slice(compat::local_id::x());

  Tensor tSgS = thr_copy_load.partition_S(gS);
  Tensor tSsS = thr_copy_load.partition_D(sS);
  //
  Tensor tDsD = thr_copy_store.partition_S(sS);
  Tensor tDgD = thr_copy_store.partition_D(gD);

  copy(tiled_copy_load, tSgS, tSsS);

  cp_async_fence();
  cp_async_wait<0>();
  syncthreads();
  //
  copy(tiled_copy_store, tDsD, tDgD);
}

template <typename Element> void copy_smem(TransposeParams<Element> params) {

  using namespace cute;

  //
  // Make tensors
  //
  auto tensor_shape = make_shape(params.M, params.N);
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape, LayoutRight{});
  Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);
  Tensor tensor_D = make_tensor(make_gmem_ptr(params.output), gmemLayoutD);

  //
  // Tile tensors
  using bM = Int<1>;
  using bN = Int<8192>;

  auto block_shape = make_shape(bM{}, bN{}); // (bM, bN)

  auto smem_layout = make_layout(block_shape, LayoutRight{});

  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
  Tensor tiled_tensor_D =
      tiled_divide(tensor_D, block_shape); // ((bN, bM), n', m')

  auto threadLayout =
      make_layout(make_shape(Int<1>{}, Int<1024>{}), LayoutRight{});

  //
  // Determine grid and block dimensions
  //

  dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(threadLayout)); // 256 threads

  constexpr int smem_size =
      int(sizeof(SharedStorageCopy<Element, decltype(smem_layout)>));

  //
  // Launch the kernel
  //
  compat::launch<
      copySmemKernel<decltype(tiled_tensor_S), decltype(tiled_tensor_D),
                     decltype(threadLayout), decltype(smem_layout), smem_size>>(
      gridDim, blockDim, tiled_tensor_S, tiled_tensor_D, threadLayout,
      smem_layout);
}
