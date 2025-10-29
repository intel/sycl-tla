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

template <class TensorS, class TensorD, class ThreadLayoutS,
          class ThreadLayoutD>
void transposeKernelNaive(TensorS const S, TensorD const DT,
                          ThreadLayoutS const tS, ThreadLayoutD const tD) {
  using namespace cute;
  using Element = typename TensorS::value_type;

  Tensor gS = S(make_coord(_, _), compat::work_group_id::x(),
                compat::work_group_id::y()); // (bM, bN)
  Tensor gDT = DT(make_coord(_, _), compat::work_group_id::x(),
                  compat::work_group_id::y()); // (bM, bN)

  Tensor tSgS = local_partition(gS, ThreadLayoutS{},
                                compat::local_id::x()); // (ThrValM, ThrValN)
  Tensor tDgDT = local_partition(gDT, ThreadLayoutD{}, compat::local_id::x());

  Tensor rmem = make_tensor_like(tSgS);

  copy(tSgS, rmem);
  copy(rmem, tDgDT);
}

template <typename Element>
void transpose_naive(TransposeParams<Element> params) {

  using namespace cute;
  //
  // Make Tensors
  //
  auto tensor_shape = make_shape(params.M, params.N);
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);

  // Make a transposed view of the output
  auto gmemLayoutDT = make_layout(tensor_shape, GenColMajor{});
  Tensor tensor_DT = make_tensor(make_gmem_ptr(params.output), gmemLayoutDT);

  //
  // Tile tensors
  //

  using bM = Int<8>;
  using bN = Int<512>;

  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)

  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
  Tensor tiled_tensor_DT =
      tiled_divide(tensor_DT, block_shape); // ((bM, bN), m', n')

  auto threadLayoutS =
      make_layout(make_shape(Int<8>{}, Int<64>{}), LayoutRight{});
  auto threadLayoutD =
      make_layout(make_shape(Int<8>{}, Int<64>{}), LayoutRight{});

  auto gridDim = compat::dim3(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
  auto blockDim = compat::dim3(size(threadLayoutS));

  //
  // Launch the kernel
  //
  compat::launch<
      transposeKernelNaive<decltype(tiled_tensor_S), decltype(tiled_tensor_DT),
                           decltype(threadLayoutS), decltype(threadLayoutD)>>(
      gridDim, blockDim, tiled_tensor_S, tiled_tensor_DT, threadLayoutS,
      threadLayoutD);
};
