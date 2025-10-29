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

// copy kernel adapted from
// https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/tiled_copy.cu

#include <cute/util/compat.hpp>
#include <sycl/sycl.hpp>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "util.h"

#include <iomanip>

template <class TensorS, class TensorD, class ThreadLayout>
void copy_kernel(TensorS S, TensorD D, ThreadLayout) {
  using namespace cute;

  // Slice the tiled tensors
  Tensor tile_S = S(make_coord(_, _), compat::work_group_id::x(),
                    compat::work_group_id::y()); // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_, _), compat::work_group_id::x(),
                    compat::work_group_id::y()); // (BlockShape_M, BlockShape_N)

  // Construct a partitioning of the tile among threads with the given thread
  // arrangement.

  // Concept:                         Tensor  ThrLayout       ThrIndex
  Tensor thr_tile_S = local_partition(
      tile_S, ThreadLayout{}, compat::local_id::x()); // (ThrValM, ThrValN)
  Tensor thr_tile_D = local_partition(
      tile_D, ThreadLayout{}, compat::local_id::x()); // (ThrValM, ThrValN)
                                                      //

  // Construct a register-backed Tensor with the same shape as each thread's
  // partition Use make_tensor to try to match the layout of thr_tile_S
  Tensor fragment = make_tensor_like(thr_tile_S); // (ThrValM, ThrValN)

  // Copy from GMEM to RMEM and from RMEM to GMEM
  copy(thr_tile_S, fragment);
  copy(fragment, thr_tile_D);
}

template <typename Element> void copy_direct(TransposeParams<Element> params) {
  //
  // Given a 2D shape, perform an efficient copy
  //

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
  //

  // Define a statically sized block (M, N).
  // Note, by convention, capital letters are used to represent static modes.
  auto block_shape = make_shape(Int<1>{}, Int<16384>{});

  if ((size<0>(tensor_shape) % size<0>(block_shape)) ||
      (size<1>(tensor_shape) % size<1>(block_shape))) {
    std::cerr << "The tensor shape must be divisible by the block shape."
              << std::endl;
  }
  // Equivalent check to the above
  if (not evenly_divides(tensor_shape, block_shape)) {
    std::cerr << "Expected the block_shape to evenly divide the tensor shape."
              << std::endl;
  }

  // Tile the tensor (m, n) ==> ((M, N), m', n') where (M, N) is the static tile
  // shape, and modes (m', n') correspond to the number of tiles.
  //
  // These will be used to determine the CUDA kernel grid dimensions.
  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape); // ((M, N), m', n')
  Tensor tiled_tensor_D =
      tiled_divide(tensor_D, block_shape); // ((M, N), m', n')

  // Thread arrangement
  Layout thr_layout =
      make_layout(make_shape(Int<1>{}, Int<1024>{}), LayoutRight{});

  //
  // Determine grid and block dimensions
  //

  auto gridDim = compat::dim3(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
  auto blockDim = compat::dim3(size(thr_layout));

  //
  // Launch the kernel
  //
  compat::launch<copy_kernel<decltype(tiled_tensor_S), decltype(tiled_tensor_D),
                             decltype(thr_layout)>>(
      gridDim, blockDim, tiled_tensor_S, tiled_tensor_D, thr_layout);
}
