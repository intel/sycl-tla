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
#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/util/print_error.hpp"
#include "util.h"

template <class TensorS, class TensorD, class BlockShape, class BlockShapeTrans,
          class ThreadLayout>
void block2DTransposedLoadKernel(TensorS const S, TensorD const DT,
                                 BlockShape const block_shape,
                                 BlockShapeTrans const block_shape_transposed,
                                 ThreadLayout const thread_layout) {
  using namespace cute;
  using Element = typename TensorS::value_type;

  /* get workgroup and local ids */
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m = int(item.get_group(0));
  auto wg_n = int(item.get_group(1));
  auto local_id = int(item.get_local_id(0));

  /* proxy coordinate tensor */
  Tensor cS = make_identity_tensor(S.shape());   // (M,N)
  Tensor cDT = make_identity_tensor(DT.shape()); // (N,M)

  auto wg_coord = make_coord(wg_m, wg_n);
  auto wg_coord_transposed = make_coord(wg_n, wg_m);

  // Tensor data = ... // (  M,  N) Tensor cta_data = local_tile(data,
  // Shape<16, 16>{}, make_coord(blockIdx.x,blockIdx.y));  // (_32,_64)
  Tensor gS = local_tile(cS, block_shape, wg_coord); // (BLK_M,BLK_N)
  Tensor gDT = local_tile(cDT, block_shape_transposed,
                          wg_coord_transposed); // (BLK_N,BLK_M);

  constexpr int CopyBits = sizeof_bits_v<Element>;
  auto transposed_load_op = XE_LOAD_2D_TRANSPOSE<CopyBits, 8, 8>{};
  auto store_op = XE_STORE_2D<CopyBits, 8, 8>{};

  /* Slice TiledCopy operations to thread (work-item) level */
  auto transpose_S = make_block_2d_copy(transposed_load_op, S);
  auto thr_transpose_S = transpose_S.get_slice(local_id);

  auto store_DT = make_block_2d_copy(store_op, DT);
  auto thr_copy_DT = store_DT.get_slice(local_id);

  /* Register fragments for transposed copy */
  auto tSrS = thr_transpose_S.partition_sg_fragment_D(gS);
  auto tDrD = thr_copy_DT.partition_sg_fragment_D(gDT);

  /* Partition global tensor (proxies) for copies */
  Tensor tSgS = thr_transpose_S.partition_S(gS);
  Tensor tDgD = thr_copy_DT.partition_D(gDT);

  // if ( cute::thread(0, 0)){
  //     print(tSgS);print("\n");
  //     print(tSrS);print("\n");
  //     print(tDgD);print("\n");
  // }

  copy(transpose_S, tSgS, tSrS);
  // copy(tSrS, tDrD);
  copy(store_DT, tSrS, tDgD);
}

class TransposeCuteName;
template <typename Element>
void block_2d_transposed_copy(TransposeParams<Element> params) {

  using namespace cute;
  //
  // Make Tensors
  //
  auto tensor_shape = make_shape(params.M, params.N);
  auto tensor_shape_trans = make_shape(params.N, params.M);
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape_trans, LayoutRight{});
  Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);
  Tensor tensor_DT = make_tensor(make_gmem_ptr(params.output), gmemLayoutD);

  // Make a transposed view of the output
  // auto gmemLayoutDT = make_layout(tensor_shape, GenColMajor{});
  // Tensor tensor_DT = make_tensor(make_gmem_ptr(params.output), gmemLayoutDT);

  sycl::queue Q;

  //
  // Tile tensors
  //

  using bM = Int<32>;
  using bN = Int<8>;

  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)
  auto block_shape_trans = make_shape(bN{}, bM{}); // (bN, bM)

  sycl::range<2> local = {bM{}, 1};
  sycl::range<2> global = {local[0] * ceil_div(shape<0>(tensor_S), bM{}),
                           local[1] * ceil_div(shape<1>(tensor_S), bN{})};

  auto threadLayout = make_layout(make_shape(bM{}, Int<1>{}), LayoutRight{});

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{syclex::sub_group_size<16>,
                                  intelex::grf_size<256>};

  auto event = Q.parallel_for<TransposeCuteName>(
      sycl::nd_range<2>(global, local), kernel_props, [=](auto) {
        block2DTransposedLoadKernel(tensor_S, tensor_DT, block_shape,
                                    block_shape_trans, threadLayout);
      });
};
