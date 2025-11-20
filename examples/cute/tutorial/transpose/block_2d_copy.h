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

#include "cute/stride.hpp"
#include "cute/swizzle_layout.hpp"
#include "cutlass/util/print_error.hpp"
#include "util.h"

template <class TensorS, class BlockShape, class TVLayout>
void block2DCopyKernel(TensorS const S,
                       BlockShape const block_shape,
                       TVLayout const sv_layout) {
  using namespace cute;
  using Element = typename TensorS::value_type;

  // get the m,n workgroup ids (thread-block indices)
  // and the local id (threadIdx.x) of this thread */
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m = int(item.get_group(0));
  auto wg_n = int(item.get_group(1));
  auto local_id = int(item.get_local_id(0));

  // create a coordinate tensor of the input matrix;
  // to be used in copy atom operations
  Tensor cS = make_identity_tensor(S.shape());   // (M,N)

  // create a wg coordinate to slice the input matrix tile
  auto wg_coord = make_coord(wg_m, wg_n);

  Tensor gS = local_tile(cS, block_shape, wg_coord); // (BLK_M,BLK_N)

  constexpr int CopyBits = sizeof_bits_v<Element>;
  constexpr int Width = 8;
  constexpr int Height = 8;

  // auto transposed_load_op = XE_LOAD_2D_TRANSPOSE<CopyBits, Width, Height>{};
  auto copy_op = XE_LOAD_2D<CopyBits, Width, Height>{};

  auto S_stride = S.stride();
  auto x_mode = find_x_mode(S_stride);
  auto y_mode = find_y_mode(S_stride);

  using CopyOp = decltype(copy_op);
  using XMode = decltype(x_mode);
  using YMode = decltype(y_mode);

  // Divide coordinate codomain into copy tiles.
  auto op_tile = Int<Width>{}  * E<XMode::value>{}
               + Int<Height>{} * E<YMode::value>{};
  auto atom_shape = shape_div(block_shape, op_tile);

  /* Slice TiledCopy operations to thread (work-item) level */
  auto copy_S = make_block_2d_copy<Element>(copy_op,
                                            S_stride,
                                            x_mode,
                                            y_mode,
                                            atom_shape,
                                            sv_layout.layout());
  auto thr_copy_S = copy_S.get_slice(local_id);

  /* Register fragments for transposed copy */
  auto tSrS = thr_copy_S.partition_sg_fragment_D(gS);

  /* Partition global tensor (proxies) for copies */
  Tensor tSgS = thr_copy_S.partition_S(gS);

  if (thread0()){
      print("block_shape: "); print(block_shape); print("\n");
      print("atom_shape: "); print(atom_shape); print("\n");

      print("S: "); print(S);print("\n");
      print("cS: "); print(cS);print("\n");
      print("gS: ");print(gS);print("\n\n");

      print("transpose_S: "); print(copy_S);print("\n\n");
      print("thr_transpose_S: "); print(thr_copy_S);print("\n\n");
      print("tSgS: ");print(tSgS);print("\n");
      print("tSrS: "); print(tSrS);print("\n\n");
  }

  copy(copy_S, tSgS, tSrS);
}

class CopyCuteName;
template <typename Element>
void block_2d_copy(TransposeParams<Element> params) {

  using namespace cute;
  //
  // Make Tensors
  //
  auto tensor_shape = make_shape(params.M, params.N);
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);

  sycl::queue Q;

  // Tile tensors
  //

  using bM = Int<8>;
  using bN = Int<8>;

  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)

  sycl::range<2> local = {1, 16}; // 1 sub-groups; keep the subgroup contiguous in the x-axis
  sycl::range<2> global = {local[0] * ceil_div(shape<0>(tensor_S), bM{}),
                           local[1] * ceil_div(shape<1>(tensor_S), bN{})};

  // Create a mapping from (S1, V64) -> (M8, N8)
  // Each sub-group owns the copy for 64 elements through the 8x8 vectorized copy atom
  // Here we just have 1 sub-group to perform a copy of a 8x8 block where each work-item
  // is responsible to copy 4 elements
  constexpr int SubgroupSize = 16;
  // constexpr int NumSubgroups = get<0>(block_shape);  // 1
  constexpr int NumSubgroups = 1;

  // In NVIDIA-CuTe, we use the tv-layout to index into the thread-mode to
  // achieve the mapping from thread idx to all the linear value indices that the
  // thread is responsible for.
  // Similarly, we aim to achieve a mapping from subgroup idx to linear value indices.
  // Contiguous thread index within a subgroup maps to mode-1 (or x-axis) because the
  // input layout is contiguous on mode-1 for coalesced load-store
  // SGV layout map subgroup & value index to (8, 8) logical tile
  auto sg_shape = make_shape(Int<NumSubgroups>{}, _1{});
  using sg_layout = decltype(make_layout(sg_shape, Stride<_1, _0>{}));
  using val_layout = decltype(Layout<Shape<_8, _8>, Stride<_8,_1>>{});
  // Layout for subgroups tiling the workgroup tile

  // Following was taken from a make_tiled_copy overload that computes
  // the equivalent of this in CuTe DSL
  // tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

  // Take the raked_products to compute the Layout_MN
  // (M,N) -> (thr_idx, val_idx)
  auto layout_mn = raked_product(sg_layout{}, val_layout{});
  // (thr_idx, val_idx) -> (M,N)
  auto sv_layout = right_inverse(layout_mn).with_shape(make_shape(size(sg_layout{}), size(val_layout{})));

#if 1
print("sg_layout: "); print(sg_layout{}); print("\n");
print("val_layout: "); print(val_layout{}); print("\n");
print("layout_mn : "); print(layout_mn);  print("\n");
print("sv_layout: "); print(sv_layout); print("\n");
#endif

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{syclex::sub_group_size<16>,
                                  intelex::grf_size<256>};

  auto event = Q.parallel_for<CopyCuteName>(
      sycl::nd_range<2>(global, local), kernel_props, [=](auto) {
        block2DCopyKernel(tensor_S, block_shape,
            sv_layout);
      });
};
