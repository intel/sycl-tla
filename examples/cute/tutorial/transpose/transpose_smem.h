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
template <class Element, class SmemLayout> struct SharedStorageTranspose {
  cute::array_aligned<Element, cute::cosize_v<SmemLayout>,
                      cutlass::detail::alignment_for_swizzle(SmemLayout{})>
      smem;
};

template <class TensorS, class TensorD, class SmemLayoutS, class ThreadLayoutS,
          class SmemLayoutD, class ThreadLayoutD, int smem_size>
void transposeSmemKernel(TensorS const S, TensorD const D,
                         SmemLayoutS const smemLayoutS, ThreadLayoutS const tS,
                         SmemLayoutD const smemLayoutD,
                         ThreadLayoutD const tD) {
  using namespace cute;
  using Element = typename TensorS::value_type;

  // Use Shared Storage structure to allocate aligned SMEM addresses.
  using SharedStorage = SharedStorageTranspose<Element, SmemLayoutD>;
  auto smem = compat::local_mem<Element[smem_size]>();
  SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem);

  // two different views of smem
  Tensor sS = make_tensor(make_smem_ptr(shared_storage.smem.data()),
                          smemLayoutS); // (bM, bN)
  Tensor sD = make_tensor(make_smem_ptr(shared_storage.smem.data()),
                          smemLayoutD); // (bN, bM)

  Tensor gS = S(make_coord(_, _), compat::work_group_id::x(),
                compat::work_group_id::y()); // (bM, bN)
  Tensor gD = D(make_coord(_, _), compat::work_group_id::y(),
                compat::work_group_id::x()); // (bN, bM)

  Tensor tSgS =
      local_partition(gS, tS, compat::local_id::x()); // (ThrValM, ThrValN)
  Tensor tSsS =
      local_partition(sS, tS, compat::local_id::x()); // (ThrValM, ThrValN)
  Tensor tDgD = local_partition(gD, tD, compat::local_id::x());
  Tensor tDsD = local_partition(sD, tD, compat::local_id::x());

  cute::copy(tSgS, tSsS); // LDGSTS

  cp_async_fence();
  cp_async_wait<0>();
  syncthreads();

  cute::copy(tDsD, tDgD);
}

template <typename Element, bool isSwizzled = false>
void transpose_smem(TransposeParams<Element> params) {

  using namespace cute;

  //
  // Make tensors
  //
  auto tensor_shape = make_shape(params.M, params.N);
  auto tensor_shape_trans = make_shape(params.N, params.M);
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape_trans, LayoutRight{});
  Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);
  Tensor tensor_D = make_tensor(make_gmem_ptr(params.output), gmemLayoutD);

  //
  // Tile tensors
  //

  using bM = Int<64>;
  using bN = Int<128>;

  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)
  auto block_shape_trans = make_shape(bN{}, bM{}); // (bN, bM)

  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
  Tensor tiled_tensor_D =
      tiled_divide(tensor_D, block_shape_trans); // ((bN, bM), n', m')

  auto tileShapeS = make_layout(block_shape, LayoutRight{});
  auto tileShapeD = make_layout(block_shape_trans, LayoutRight{});

  auto smemLayoutS = tileShapeS;
  auto smemLayoutD = composition(smemLayoutS, tileShapeD);
  auto smemLayoutS_swizzle = composition(Swizzle<5, 0, 5>{}, tileShapeS);
  auto smemLayoutD_swizzle = composition(smemLayoutS_swizzle, tileShapeD);

  auto threadLayoutS =
      make_layout(make_shape(Int<8>{}, Int<64>{}), LayoutRight{});
  auto threadLayoutD =
      make_layout(make_shape(Int<8>{}, Int<64>{}), LayoutRight{});

  constexpr int smem_size =
      int(sizeof(SharedStorageTranspose<Element, decltype(smemLayoutS)>));

  //
  // Determine grid and block dimensions
  //

  dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(threadLayoutS)); // 256 threads

  if constexpr (isSwizzled) {
    compat::launch<transposeSmemKernel<
        decltype(tiled_tensor_S), decltype(tiled_tensor_D),
        decltype(smemLayoutS_swizzle), decltype(threadLayoutS),
        decltype(smemLayoutD_swizzle), decltype(threadLayoutD), smem_size>>(
        gridDim, blockDim, tiled_tensor_S, tiled_tensor_D, smemLayoutS_swizzle,
        threadLayoutS, smemLayoutD_swizzle, threadLayoutD);
  } else {
    compat::launch<transposeSmemKernel<
        decltype(tiled_tensor_S), decltype(tiled_tensor_D),
        decltype(smemLayoutS), decltype(threadLayoutS), decltype(smemLayoutD),
        decltype(threadLayoutD), smem_size>>(
        gridDim, blockDim, tiled_tensor_S, tiled_tensor_D, smemLayoutS,
        threadLayoutS, smemLayoutD, threadLayoutD);
  }
}
