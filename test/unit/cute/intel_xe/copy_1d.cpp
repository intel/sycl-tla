/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
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

#include "cutlass/detail/layout.hpp"

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>

#include "cutlass_unit_test.h"

using namespace cute;
using namespace cutlass;
using namespace compat::experimental;

#define SUBGROUP_SIZE (16)

template<class...> class CopyKernelVectorizedName;

template <class TensorS, class TensorD>
void copy_kernel_vectorized(TensorS tile_S, TensorD tile_D) {
  using namespace cute;

  using Element = typename TensorS::value_type;

  // Shared memory buffers
  auto smem = compat::local_mem<Element[size(tile_S)]>();
  Tensor sTensor = make_tensor(make_smem_ptr(smem), tile_S.layout());

  using ElementUint = typename uint_bit<sizeof_bits_v<Element>>::type;

  // GMEM access type: use uint64_t for 8-bit elements (max 8 elements per vector),
  // uint128_t for wider types
  using GmemAccessType = conditional_t<(sizeof(Element) < 2), uint64_t, cutlass::uint128_t>;

  // GMEM copy atoms: vectorized 1D load/store (group_load / group_store)
  using traits_load = Copy_Traits<XE_1D_LOAD_GLOBAL<ElementUint, GmemAccessType>>;
  using Atom_load = Copy_Atom<traits_load, Element>;
  using traits_store = Copy_Traits<XE_1D_STORE_GLOBAL<GmemAccessType, ElementUint>>;
  using Atom_store = Copy_Atom<traits_store, Element>;

  // SLM copy atoms: per-lane scatter load/store with S == D for symmetric transfer
  using traits_ldsm = Copy_Traits<XE_1D_LDSM<ElementUint, ElementUint>>;
  using Atom_ldsm = Copy_Atom<traits_ldsm, Element>;
  using traits_stsm = Copy_Traits<XE_1D_STSM<ElementUint, ElementUint>>;
  using Atom_stsm = Copy_Atom<traits_stsm, Element>;

  // GMEM tiled copy: 16 threads, each loading GmemAccessType worth of elements
  auto GmemVecLayout = make_layout(
      make_shape(_1{}, Int<sizeof(GmemAccessType) / sizeof(Element)>{}),
      Stride<Int<sizeof(GmemAccessType) / sizeof(Element)>, _1>{});
  auto ThreadLayout = make_layout(make_shape(_1{}, _16{}));
  auto tiled_copy_load = make_tiled_copy(Atom_load{}, ThreadLayout, GmemVecLayout);
  auto tiled_copy_store = make_tiled_copy(Atom_store{}, ThreadLayout, GmemVecLayout);

  // SLM tiled copy: 16 threads, 1 element per atom (symmetric S==D)
  auto SlmVecLayout = make_layout(make_shape(_1{}, _1{}));
  auto tiled_ldsm = make_tiled_copy(Atom_ldsm{}, ThreadLayout, SlmVecLayout);
  auto tiled_stsm = make_tiled_copy(Atom_stsm{}, ThreadLayout, SlmVecLayout);

  // Partition for GMEM
  auto thr_copy_load = tiled_copy_load.get_thread_slice(ThreadIdxX());
  auto thr_copy_store = tiled_copy_store.get_thread_slice(ThreadIdxX());

  Tensor thr_tile_load_S = thr_copy_load.partition_S(tile_S);
  Tensor thr_tile_store_D = thr_copy_store.partition_D(tile_D);

  Tensor fragment = make_fragment_like(thr_copy_load.partition_D(tile_S));

  // Partition for SLM
  auto thr_copy_ldsm = tiled_ldsm.get_thread_slice(ThreadIdxX());
  auto thr_copy_stsm = tiled_stsm.get_thread_slice(ThreadIdxX());

  Tensor thr_tile_ldsm_S = thr_copy_ldsm.partition_S(sTensor);
  Tensor thr_tile_stsm_D = thr_copy_stsm.partition_D(sTensor);

  Tensor slm_frag_load = make_fragment_like(thr_copy_ldsm.partition_D(sTensor));
  Tensor slm_frag_store = make_fragment_like(thr_copy_stsm.partition_S(sTensor));

  // Copy: GMEM -> registers
  prefetch(tiled_copy_load, thr_tile_load_S);
  copy(tiled_copy_load, thr_tile_load_S, fragment);

  // Copy: registers -> SLM via STSM (flatten and copy element-wise to slm_frag_store, then STSM)
  auto flat_frag = make_tensor(fragment.data(), make_layout(size(fragment)));
  auto flat_slm_store = make_tensor(slm_frag_store.data(), make_layout(size(slm_frag_store)));
  CUTE_UNROLL
  for (int i = 0; i < size(flat_frag); ++i) {
    flat_slm_store(i) = flat_frag(i);
  }
  copy(tiled_stsm, slm_frag_store, thr_tile_stsm_D);

  // Clear registers
  clear(fragment);
  clear(slm_frag_load);

  // Copy: SLM -> registers via LDSM
  copy(tiled_ldsm, thr_tile_ldsm_S, slm_frag_load);

  // Copy: slm_frag_load -> fragment (flatten and copy element-wise)
  auto flat_slm_load = make_tensor(slm_frag_load.data(), make_layout(size(slm_frag_load)));
  auto flat_frag2 = make_tensor(fragment.data(), make_layout(size(fragment)));
  CUTE_UNROLL
  for (int i = 0; i < size(flat_frag2); ++i) {
    flat_frag2(i) = flat_slm_load(i);
  }

  // Copy: registers -> GMEM
  copy(tiled_copy_store, fragment, thr_tile_store_D);
}

TEST(PVC_1d_copy, copy_double) {
  // Test 64-bit (double)
  {
    constexpr int M = 1;
    constexpr int N = 128;
    using Element = double;

    cutlass::host_vector<Element> host_src(M * N);
    cutlass::host_vector<Element> host_output(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<Element>(i);
    }

    cutlass::device_vector<Element> device_src = host_src;
    cutlass::device_vector<Element> device_output(M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src.data()),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output.data()),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    static constexpr auto subgroup_size = 16;
    auto blockDim = compat::dim3(subgroup_size);

    launch<copy_kernel_vectorized<decltype(S), decltype(D)>, CopyKernelVectorizedName<decltype(S), decltype(D)>>(
        launch_policy{
            compat::dim3(1), blockDim,
            kernel_properties{sycl_exp::sub_group_size<SUBGROUP_SIZE>}},
        S, D);

    compat::wait_and_throw();
    host_output = device_output;
    for (int i = 0; i < M * N; ++i) {
      EXPECT_EQ(host_output[i], host_src[i]);
    }
  }

  // Test 32-bit (float)
  {
    constexpr int M = 1;
    constexpr int N = 128;
    using Element = float;

    cutlass::host_vector<Element> host_src(M * N);
    cutlass::host_vector<Element> host_output(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<Element>(i);
    }

    cutlass::device_vector<Element> device_src = host_src;
    cutlass::device_vector<Element> device_output(M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src.data()),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output.data()),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    static constexpr auto subgroup_size = 16;
    auto blockDim = compat::dim3(subgroup_size);

    launch<copy_kernel_vectorized<decltype(S), decltype(D)>, CopyKernelVectorizedName<decltype(S), decltype(D)>>(
        launch_policy{
            compat::dim3(1), blockDim,
            kernel_properties{sycl_exp::sub_group_size<SUBGROUP_SIZE>}},
        S, D);

    compat::wait_and_throw();
    host_output = device_output;
    for (int i = 0; i < M * N; ++i) {
      EXPECT_EQ(host_output[i], host_src[i]);
    }
  }

  // Test 16-bit (uint16_t)
  {
    constexpr int M = 1;
    constexpr int N = 128;
    using Element = uint16_t;

    cutlass::host_vector<Element> host_src(M * N);
    cutlass::host_vector<Element> host_output(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<Element>(i);
    }

    cutlass::device_vector<Element> device_src = host_src;
    cutlass::device_vector<Element> device_output(M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src.data()),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output.data()),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    static constexpr auto subgroup_size = 16;
    auto blockDim = compat::dim3(subgroup_size);

    launch<copy_kernel_vectorized<decltype(S), decltype(D)>, CopyKernelVectorizedName<decltype(S), decltype(D)>>(
        launch_policy{
            compat::dim3(1), blockDim,
            kernel_properties{sycl_exp::sub_group_size<SUBGROUP_SIZE>}},
        S, D);

    compat::wait_and_throw();
    host_output = device_output;
    for (int i = 0; i < M * N; ++i) {
      EXPECT_EQ(host_output[i], host_src[i]);
    }
  }

  // Test 8-bit (uint8_t)
  {
    constexpr int M = 1;
    constexpr int N = 256;
    using Element = uint8_t;

    cutlass::host_vector<Element> host_src(M * N);
    cutlass::host_vector<Element> host_output(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<Element>(i & 0xFF);
    }

    cutlass::device_vector<Element> device_src = host_src;
    cutlass::device_vector<Element> device_output(M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src.data()),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output.data()),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    static constexpr auto subgroup_size = 16;
    auto blockDim = compat::dim3(subgroup_size);

    launch<copy_kernel_vectorized<decltype(S), decltype(D)>, CopyKernelVectorizedName<decltype(S), decltype(D)>>(
        launch_policy{
            compat::dim3(1), blockDim,
            kernel_properties{sycl_exp::sub_group_size<SUBGROUP_SIZE>}},
        S, D);

    compat::wait_and_throw();
    host_output = device_output;
    for (int i = 0; i < M * N; ++i) {
      EXPECT_EQ(host_output[i], host_src[i]);
    }
  }
}