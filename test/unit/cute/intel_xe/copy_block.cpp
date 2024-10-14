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

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>
#include <syclcompat.hpp>

#include "cutlass_unit_test.h"

using namespace cute;
using namespace cutlass;

template <class TensorS, class TensorD, class TiledLoad, class TiledStore, class CopyOp = void>
void copy_kernel_vectorized(TensorS S, TensorD D, TiledLoad load,
                            TiledStore store) {
  const int m_coord = 0;
  const int n_coord = 0;
  const int l_coord = BlockIdxZ();

  // ==========  load   ==========
  auto thr_copy_load = load.get_thread_slice(syclcompat::local_id::x());
  auto thr_tile_load_D = thr_copy_load.partition_D(S);
  auto fragment = make_fragment_like(thr_tile_load_D);
  auto ld_tensor =
      load.get_pvc_tensor(make_coord(m_coord, n_coord, l_coord),
                          fragment.shape(), typename TiledLoad::Shape_MN{});
  if constexpr (cute::detail::has_prefetch<CopyOp>) prefetch(load, ld_tensor);
  copy(load, ld_tensor, fragment);

  // ==========  store   ==========
  auto thr_copy_store = store.get_thread_slice(syclcompat::local_id::x());
  Tensor frag_view =
      make_tensor(static_cast<decltype(fragment) &&>(fragment).data(),
                  thr_copy_store.partition_S(D).shape());
  auto st_tensor = store.get_pvc_tensor(make_coord(m_coord, n_coord, l_coord),
                                        frag_view.shape(),
                                        typename TiledStore::Shape_MN{});
  copy(store, frag_view, st_tensor);

#if 1
  if (thread(1)) {
    print("fragment: ");
    print(fragment.layout());
    print("\n");

    print("ld_tensor: ");
    print(ld_tensor.layout());
    print("\n");

    print("frag_view: ");
    print(frag_view.layout());
    print("\n");

    print("st_tensor: ");
    print(st_tensor.layout());
    print("\n");
  }
#endif
}

TEST(PVC_2d_copy, load_store) {
  {
    print("XE_2D_U16x1x16_LD_N test: \n");
    constexpr int M = 1;
    constexpr int N = 16;
    using dtype = uint16_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x1x16_LD_N>, uint16_t>{}.with(device_src, N, M,
                                                              N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_1, _1>, Stride<_1, _0>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x1x16_ST_N>, uint16_t>{}.with(device_output, N,
                                                              M, N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_1, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U16x1x16_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U16x1x16_LD_N test end \n\n");
  }

  {
    print("XE_2D_U16x2x16_LD_N test: \n");
    constexpr int M = 2;
    constexpr int N = 16;
    using dtype = uint16_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x2x16_LD_N>, uint16_t>{}.with(device_src, N, M,
                                                              N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_2, _1>, Stride<_1, _0>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x2x16_ST_N>, uint16_t>{}.with(device_output, N,
                                                              M, N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_2, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U16x2x16_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U16x2x16_LD_N test end \n\n");
  }

  {
    print("XE_2D_U16x4x16_LD_N test: \n");
    constexpr int M = 4;
    constexpr int N = 16;
    using dtype = uint16_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x4x16_LD_N>, uint16_t>{}.with(device_src, N, M,
                                                              N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_4, _1>, Stride<_1, _0>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x4x16_ST_N>, uint16_t>{}.with(device_output, N,
                                                              M, N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_4, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U16x4x16_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U16x4x16_LD_N test end \n\n");
  }

  {
    print("XE_2D_U16x8x16_LD_N test: \n");
    constexpr int M = 8;
    constexpr int N = 16;
    using dtype = uint16_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x8x16_LD_N>, uint16_t>{}.with(device_src, N, M,
                                                              N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_8, _1>, Stride<_1, _0>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N>, uint16_t>{}.with(device_output, N,
                                                              M, N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_8, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U16x8x16_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U16x8x16_LD_N test end \n\n");
  }

  {
    print("XE_2D_U16x16x16_LD_N test: \n");
    constexpr int M = 16;
    constexpr int N = 16;
    using dtype = uint16_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_N>, uint16_t>{}.with(device_src, N, M,
                                                               N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_16, _1>, Stride<_1, _0>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N>, uint16_t>{}.with(device_output, N,
                                                              M, N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_8, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U16x16x16_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U16x16x16_LD_N test end \n\n");
  }

  {
    print("XE_2D_U16x16x16_LD_N test: \n");
    constexpr int M = 32;
    constexpr int N = 16;
    using dtype = uint16_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_N>, uint16_t>{}.with(device_src, N, M,
                                                               N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_16, _1>, Stride<_1, _0>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N>, uint16_t>{}.with(device_output, N,
                                                              M, N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_8, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U16x16x16_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U16x16x16_LD_N test end \n\n");
  }

  {
    print("XE_2D_U16x1x32_LD_N test: \n");
    constexpr int M = 1;
    constexpr int N = 32;
    using dtype = uint16_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D = make_tensor(
        make_gmem_ptr(device_output),
        make_layout(Shape<Int<2>, Int<16>>{}, Stride<Int<16>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x1x32_LD_N>, uint16_t>{}.with(device_src, N, M,
                                                              N),
        Layout<Shape<_1, _16>>{}, Layout<Shape<_1, _2>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x2x16_ST_N>, uint16_t>{}.with(device_output, 16,
                                                              2, 16),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_2, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U16x1x32_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U16x1x32_LD_N test end \n\n");
  }

  {
    print("XE_2D_U16x2x32_LD_N test: \n");
    constexpr int M = 2;
    constexpr int N = 32;
    using dtype = uint16_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D = make_tensor(
        make_gmem_ptr(device_output),
        make_layout(Shape<Int<4>, Int<16>>{}, Stride<Int<16>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x2x32_LD_N>, uint16_t>{}.with(device_src, N, M,
                                                              N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_2, _2>, Stride<_1, _2>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x4x16_ST_N>, uint16_t>{}.with(device_output, 16,
                                                              4, 16),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_4, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U16x2x32_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    /* 0  1  2  3 ... 16 ... 31
       32 33 34 35... 48 ... 64
       thread 0
       0 32 16 48
       0   1 ...
       32 33 ...
       16 17 ...
       48 49 ...
    */
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 16; ++j) {
        // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
        EXPECT_EQ(host_output[i * 16 + j],
                  host_src[(i % 2) * 32 + j + (i / 2) * 16]);
      }
    }
    print("XE_2D_U16x2x32_LD_N test end \n\n");
  }

  {
    print("XE_2D_U16x4x32_LD_N test: \n");
    constexpr int M = 4;
    constexpr int N = 32;
    using dtype = uint16_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D = make_tensor(
        make_gmem_ptr(device_output),
        make_layout(Shape<Int<8>, Int<16>>{}, Stride<Int<16>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x4x32_LD_N>, uint16_t>{}.with(device_src, N, M,
                                                              N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_4, _2>, Stride<_1, _4>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N>, uint16_t>{}.with(device_output, 16,
                                                              8, 16),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_8, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U16x4x32_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);

    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 16; ++j) {
        // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
        EXPECT_EQ(host_output[i * 16 + j],
                  host_src[(i % 4) * 32 + j + (i / 4) * 16]);
      }
    }
    print("XE_2D_U16x4x32_LD_N test end \n\n");
  }

  {
    print("XE_2D_U16x8x32_LD_N test: \n");
    constexpr int M = 8;
    constexpr int N = 32;
    using dtype = uint16_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D = make_tensor(
        make_gmem_ptr(device_output),
        make_layout(Shape<Int<16>, Int<16>>{}, Stride<Int<16>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x8x32_LD_N>, uint16_t>{}.with(device_src, N, M,
                                                              N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_8, _2>, Stride<_1, _8>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N>, uint16_t>{}.with(device_output, 16,
                                                              16, 16),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_8, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U16x8x32_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);

    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 16; ++j) {
        // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
        EXPECT_EQ(host_output[i * 16 + j],
                  host_src[(i % 8) * 32 + j + (i / 8) * 16]);
      }
    }
    print("XE_2D_U16x8x32_LD_N test end \n\n");
  }

  {
    print("XE_2D_U16x16x32_LD_N test: \n");
    constexpr int M = 16;
    constexpr int N = 32;
    using dtype = uint16_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D = make_tensor(
        make_gmem_ptr(device_output),
        make_layout(Shape<Int<32>, Int<16>>{}, Stride<Int<16>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x16x32_LD_N>, uint16_t>{}.with(device_src, N, M,
                                                               N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_16, _2>, Stride<_1, _16>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N>, uint16_t>{}.with(device_output, 16,
                                                              32, 16),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_8, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U16x16x32_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);

    for (int i = 0; i < 32; ++i) {
      for (int j = 0; j < 16; ++j) {
        // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
        EXPECT_EQ(host_output[i * 16 + j],
                  host_src[(i % 16) * 32 + j + (i / 16) * 16]);
      }
    }
    print("XE_2D_U16x16x32_LD_N test end \n\n");
  }

  {
    print("XE_2D_U16x32x32_LD_N test: \n");
    constexpr int M = 32;
    constexpr int N = 32;
    using dtype = uint16_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D = make_tensor(
        make_gmem_ptr(device_output),
        make_layout(Shape<Int<64>, Int<16>>{}, Stride<Int<16>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N>, uint16_t>{}.with(device_src, N, M,
                                                               N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_32, _2>, Stride<_1, _32>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N>, uint16_t>{}.with(device_output, 16,
                                                              64, 16),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_8, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U16x32x32_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);

    for (int i = 0; i < 64; ++i) {
      for (int j = 0; j < 16; ++j) {
        // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
        EXPECT_EQ(host_output[i * 16 + j],
                  host_src[(i % 32) * 32 + j + (i / 32) * 16]);
      }
    }
    print("XE_2D_U16x32x32_LD_N test end \n\n");
  }

  {
    print("XE_2D_U32x1x16_LD_N test: \n");
    constexpr int M = 1;
    constexpr int N = 16;
    using dtype = uint32_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x1x16_LD_N>, dtype>{}.with(device_src, N, M, N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_1, _1>, Stride<_1, _0>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x1x16_ST_N>, dtype>{}.with(device_output, N, M,
                                                           N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_1, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U32x1x16_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U32x1x16_LD_N test end \n\n");
  }

  {
    print("XE_2D_U32x2x16_LD_N test: \n");
    constexpr int M = 2;
    constexpr int N = 16;
    using dtype = uint32_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x2x16_LD_N>, dtype>{}.with(device_src, N, M, N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_2, _1>, Stride<_1, _0>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x2x16_ST_N>, dtype>{}.with(device_output, N, M,
                                                           N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_2, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U32x2x16_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U32x2x16_LD_N test end \n\n");
  }

  {
    print("XE_2D_U32x4x16_LD_N test: \n");
    constexpr int M = 4;
    constexpr int N = 16;
    using dtype = uint32_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x4x16_LD_N>, dtype>{}.with(device_src, N, M, N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_4, _1>, Stride<_1, _0>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x4x16_ST_N>, dtype>{}.with(device_output, N, M,
                                                           N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_4, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U32x4x16_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U32x4x16_LD_N test end \n\n");
  }

  {
    print("XE_2D_U32x8x16_LD_N test: \n");
    constexpr int M = 8;
    constexpr int N = 16;
    using dtype = uint32_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x8x16_LD_N>, dtype>{}.with(device_src, N, M, N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_8, _1>, Stride<_1, _0>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x8x16_ST_N>, dtype>{}.with(device_output, N, M,
                                                           N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_8, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U32x8x16_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U32x8x16_LD_N test end \n\n");
  }

  {
    print("XE_2D_U32x16x16_LD_N test: \n");
    constexpr int M = 16;
    constexpr int N = 16;
    using dtype = uint32_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x16x16_LD_N>, dtype>{}.with(device_src, N, M,
                                                            N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_16, _1>, Stride<_1, _0>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x8x16_ST_N>, dtype>{}.with(device_output, N, M,
                                                           N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_8, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U32x16x16_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U32x16x16_LD_N test end \n\n");
  }

  {
    print("XE_2D_U32x32x16_LD_N test: \n");
    constexpr int M = 32;
    constexpr int N = 16;
    using dtype = uint32_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x32x16_LD_N>, dtype>{}.with(device_src, N, M,
                                                            N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_32, _1>, Stride<_1, _0>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x8x16_ST_N>, dtype>{}.with(device_output, N, M,
                                                           N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_8, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U32x32x16_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      //  EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U32x32x16_LD_N test end \n\n");
  }

  {
    print("XE_2D_U8x1x32_LD_N test: \n");
    constexpr int M = 128;
    constexpr int N = 32;
    using dtype = char;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U8x1x32_LD_N>, dtype>{}.with(device_src, N, M, N),
        Layout<Shape<_1, _16>>{}, Layout<Shape<_1, _2>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U8x2x32_ST_N>, dtype>{}.with(device_output, N, M,
                                                          N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_2, _2>, Stride<_2, _1>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U8x1x32_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U8x1x32_LD_N test end \n\n");
  }

  {
    print("XE_2D_U8x2x32_LD_N test: \n");
    constexpr int M = 2;
    constexpr int N = 32;
    using dtype = uint8_t;

    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    auto *device_src = syclcompat::malloc<dtype>(M * N);
    auto *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U8x2x32_LD_N>, dtype>{}.with(device_src, N, M, N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_2, _2>, Stride<_1, _2>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U8x2x32_ST_N>, dtype>{}.with(device_output, N, M,
                                                          N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_2, _2>, Stride<_2, _1>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U8x2x32_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U8x2x32_LD_N test end \n\n");
  }

  {
    print("XE_2D_U8x4x32_LD_N test: \n");
    constexpr int M = 4;
    constexpr int N = 32;
    using dtype = char;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U8x4x32_LD_N>, dtype>{}.with(device_src, N, M, N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_4, _2>, Stride<_2, _1>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U8x2x32_ST_N>, dtype>{}.with(device_output, N, M,
                                                          N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_2, _2>, Stride<_2, _1>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U8x4x32_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U8x4x32_LD_N test end \n\n");
  }

  {
    print("XE_2D_U8x8x32_LD_N test: \n");
    constexpr int M = 8;
    constexpr int N = 32;
    using dtype = char;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U8x8x32_LD_N>, dtype>{}.with(device_src, N, M, N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_8, _2>, Stride<_2, _1>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U8x2x32_ST_N>, dtype>{}.with(device_output, N, M,
                                                          N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_2, _2>, Stride<_2, _1>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U8x8x32_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U8x8x32_LD_N test end \n\n");
  }

  {
    print("XE_2D_U8x16x32_LD_N test: \n");
    constexpr int M = 16;
    constexpr int N = 32;
    using dtype = char;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U8x16x32_LD_N>, dtype>{}.with(device_src, N, M, N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_16, _2>, Stride<_2, _1>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U8x2x32_ST_N>, dtype>{}.with(device_output, N, M,
                                                          N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_2, _2>, Stride<_2, _1>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U8x16x32_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U8x16x32_LD_N test end \n\n");
  }

  {
    print("XE_2D_U8x32x32_LD_N test: \n");
    constexpr int M = 32;
    constexpr int N = 32;
    using dtype = char;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U8x32x32_LD_N>, dtype>{}.with(device_src, N, M, N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_32, _2>, Stride<_2, _1>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U8x2x32_ST_N>, dtype>{}.with(device_output, N, M,
                                                          N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_2, _2>, Stride<_2, _1>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U8x32x32_LD_N>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < M * N; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U8x32x32_LD_N test end \n\n");
  }

  {
    print("XE_2D_U32x16x2_LD_T test: \n");
    constexpr int M = 16;
    constexpr int N = 2;
    using dtype = uint32_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<N>, Int<M>>{}, Stride<Int<M>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x16x2_LD_T>, dtype>{}.with(device_src, N, M, N),
        Layout<Shape<_16, _1>, Stride<_1, _0>>{},
        Layout<Shape<_1, _2>, Stride<_0, _1>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x2x16_ST_N>, dtype>{}.with(device_output, M, N,
                                                           M),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_2, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U32x16x2_LD_T>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
        EXPECT_EQ(host_output[i * M + j], host_src[j * N + i]);
      }
    }
    print("XE_2D_U32x16x2_LD_T test end \n\n");
  }

  {
    print("XE_2D_U32x16x4_LD_T test: \n");
    constexpr int M = 16;
    constexpr int N = 4;
    using dtype = uint32_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<N>, Int<M>>{}, Stride<Int<M>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x16x4_LD_T>, dtype>{}.with(device_src, N, M, N),
        Layout<Shape<_16, _1>, Stride<_1, _0>>{},
        Layout<Shape<_1, _4>, Stride<_0, _1>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x4x16_ST_N>, dtype>{}.with(device_output, M, N,
                                                           M),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_4, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U32x16x4_LD_T>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
        EXPECT_EQ(host_output[i * M + j], host_src[j * N + i]);
      }
    }
    print("XE_2D_U32x16x4_LD_T test end \n\n");
  }

  {
    print("XE_2D_U32x16x8_LD_T test: \n");
    constexpr int M = 16;
    constexpr int N = 8;
    using dtype = uint32_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<N>, Int<M>>{}, Stride<Int<M>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x16x8_LD_T>, dtype>{}.with(device_src, N, M, N),
        Layout<Shape<_16, _1>, Stride<_1, _0>>{},
        Layout<Shape<_1, _8>, Stride<_0, _1>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U32x8x16_ST_N>, dtype>{}.with(device_output, M, N,
                                                           M),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_8, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U32x16x8_LD_T>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
        EXPECT_EQ(host_output[i * M + j], host_src[j * N + i]);
      }
    }
    print("XE_2D_U32x16x8_LD_T test end \n\n");
  }

  {
    print("XE_2D_U16x16x16_LD_V test: \n");
    constexpr int M = 16;
    constexpr int N = 16;
    using dtype = uint16_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>((dtype *)device_output, host_output.data(),
                              M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_V>, dtype>{}.with(device_src, N, M,
                                                            N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_16, _1>, Stride<_1, _0>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N>, dtype>{}.with(device_output, N, M,
                                                           N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_8, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U16x16x16_LD_V>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < 64; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U16x16x16_LD_V test end \n\n");
  }

  {
    print("XE_2D_U16x32x16_LD_V test: \n");
    constexpr int M = 32;
    constexpr int N = 16;
    using dtype = uint16_t;
    //
    // Allocate and initialize
    //
    std::vector<dtype> host_src(M * N);
    std::vector<dtype> host_output(M * N);

    dtype *device_src = syclcompat::malloc<dtype>(M * N);
    dtype *device_output = syclcompat::malloc<dtype>(M * N);

    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<dtype>(i);
    }

    syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
    syclcompat::memcpy<dtype>((dtype *)device_output, host_output.data(),
                              M * N);

    Tensor S =
        make_tensor(make_gmem_ptr(device_src),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(device_output),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x32x16_LD_V>, dtype>{}.with(device_src, N, M,
                                                            N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_32, _1>, Stride<_1, _0>>{});
    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N>, dtype>{}.with(device_output, N, M,
                                                           N),
        Layout<Shape<_1, _16>, Stride<_0, _1>>{},
        Layout<Shape<_8, _1>, Stride<_1, _0>>{});
    static constexpr auto subgroup_size = 16;
    auto blockDim = syclcompat::dim3(size(tiled_load));
//
// Launch the kernel
//
    syclcompat::experimental::launch<
        copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store), XE_2D_U16x32x16_LD_V>,
        subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
    for (int i = 0; i < 64; ++i) {
      // printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
      EXPECT_EQ(host_output[i], host_src[i]);
    }
    print("XE_2D_U16x32x16_LD_V test end \n\n");
  }
}

TEST(PVC_2d_copy, load_store_XE_2D_U16x16x8_LD_T_And_XE_2D_U16x8x16_ST_N) {
  constexpr int M = 8;
  constexpr int N = 16;
  using dtype = uint16_t;

  std::vector<dtype> host_src(M * N);
  std::vector<dtype> host_output(M * N);

  dtype *device_src = syclcompat::malloc<dtype>(M * N);
  dtype *device_output = syclcompat::malloc<dtype>(M * N);

  for (size_t i = 0; i < host_src.size(); ++i) {
    host_src[i] = static_cast<dtype>(i);
  }

  syclcompat::memcpy<dtype>(device_src, host_src.data(), M * N);
  syclcompat::memcpy<dtype>(device_output, host_output.data(), M * N);

  Tensor S =
      make_tensor(make_gmem_ptr(device_src),
                  make_layout(Shape<Int<M>, Int<N>>{}, Stride<_1, Int<M>>{}));
  Tensor D =
      make_tensor(make_gmem_ptr(device_output),
                  make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

  auto tiled_load =
      make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x16x8_LD_T>, dtype>{}.with(
                          device_src, M, N, M),
                      Layout<Shape<_16, _1>, Stride<_1, _0>>{},
                      Layout<Shape<_1, _8>, Stride<_0, _1>>{});
  auto tiled_store =
      make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N>, dtype>{}.with(
                          device_output, N, M, N),
                      Layout<Shape<_1, _16>, Stride<_0, _1>>{},
                      Layout<Shape<_8, _1>, Stride<_1, _0>>{});

  static constexpr auto subgroup_size = 16;
  auto blockDim = syclcompat::dim3(size(tiled_load));

  syclcompat::experimental::launch<
      copy_kernel_vectorized<decltype(S), decltype(D), decltype(tiled_load),
                             decltype(tiled_store)>,
      subgroup_size>(1, blockDim, S, D, tiled_load, tiled_store);

  syclcompat::wait_and_throw();
  syclcompat::memcpy<dtype>(host_output.data(), device_output, M * N);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      EXPECT_EQ(host_output[i * N + j], host_src[i + j * M]);
    }
  }
}
