/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "utils.hpp"

using namespace cute;

using TileShape = Shape<_128, _128, _32>;

using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<_2,_2,_1>>,  // 2x2x1 thread group
        Tile<_32,_32,_16>>;       // 32x32x16 MMA for LDSM, 1x2x1 value group

// Smem
using SmemLayoutAtomA = decltype(
composition(Swizzle<3,3,3>{},
            Layout<Shape <_64, _8>,
                    Stride< _1,_64>>{}));
using SmemCopyAtomA = Copy_Atom<SM75_U16x8_LDSM_T, half_t>;

// Gmem
using GmemTiledCopyA = decltype(
make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half_t>{},
                Layout<Shape <_16, _8>,
                        Stride< _1,_16>>{},
                Layout<Shape < _8, _1>>{}));

// Smem
using SmemLayoutAtomB = decltype(
composition(Swizzle<3,3,3>{},
            Layout<Shape <_64, _8>,
                    Stride< _1,_64>>{}));
using SmemCopyAtomB = Copy_Atom<SM75_U16x8_LDSM_T, half_t>;

// Gmem
using GmemTiledCopyB = decltype(
make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half_t>{},
                Layout<Shape <_16, _8>,
                        Stride< _1,_16>>{},
                Layout<Shape < _8, _1>>{}));

using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{},
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}))));
using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtomB{},
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}))));

template <class MShape, class NShape, class KShape,
          class TA, class AStride,
          class TB, class BStride,
          class TC, class CStride,
          class Alpha, class Beta>
__global__ static
void
gemm_device(MShape M, NShape N, KShape K,
            TA const* A, AStride dA,
            TB const* B, BStride dB,
            TC      * C, CStride dC,
            Alpha alpha, Beta beta)
{
  using namespace cute;
  using X = Underscore;

  // Shared memory buffers
  __shared__ half_t smemA[cosize_v<SmemLayoutA>];
  __shared__ half_t smemB[cosize_v<SmemLayoutB>];
  auto sA = make_tensor(make_smem_ptr(smemA), SmemLayoutA{});               // (BLK_M,BLK_K)
  auto sB = make_tensor(make_smem_ptr(smemB), SmemLayoutB{});               // (BLK_N,BLK_K)

  // Represent the full tensors
  auto mA = make_tensor(make_gmem_ptr(A), make_shape(M,K), dA);      // (M,K)
  auto mB = make_tensor(make_gmem_ptr(B), make_shape(N,K), dB);      // (N,K)
  auto mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);      // (M,N)

  // Get the appropriate blocks for this thread block --
  // potential for thread block locality
  auto blk_shape = TileShape{};// (BLK_M,BLK_N,BLK_K)
  // Compute m_coord, n_coord, and l_coord with their post-tiled shapes
  auto m_coord = idx2crd(int(blockIdx.x), shape<0>(blk_shape));
  auto n_coord = idx2crd(int(blockIdx.y), shape<1>(blk_shape));
  auto blk_coord = make_coord(m_coord, n_coord, _);            // (m,n,k)

  auto gA = local_tile(mA, blk_shape, blk_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  auto gB = local_tile(mB, blk_shape, blk_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  auto gC = local_tile(mC, blk_shape, blk_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  //
  // Partition the copying of A and B tiles across the threads
  //

  GmemTiledCopyA gmem_tiled_copy_A;
  GmemTiledCopyB gmem_tiled_copy_B;
  auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(threadIdx.x);
  auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(threadIdx.x);

  Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
  Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
  Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
  Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

  //
  // Define C accumulators and A/B partitioning
  //

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tCrA  = thr_mma.partition_fragment_A(sA(_,_));                    // (MMA,MMA_M,MMA_K)
  Tensor tCrB  = thr_mma.partition_fragment_B(sB(_,_));                    // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);

  auto smem_tiled_copy_A   = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
  auto smem_thr_copy_A     = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
  Tensor tCsA           = smem_thr_copy_A.partition_S(sA);                   // (CPY,CPY_M,CPY_K,PIPE)
  Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

  auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
  auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(threadIdx.x);
  Tensor tCsB              = smem_thr_copy_B.partition_S(sB);                // (CPY,CPY_N,CPY_K,PIPE)
  Tensor tCrB_copy_view    = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); // (MMA,MMA_M,MMA_N)

  // Clear the accumulators
  clear(tCrC);

  int  k_tile_count = size<2>(gA);

  for (int k = 0; k < k_tile_count; ++k)
  {
    // Copy gmem to smem
    copy(tAgA(_,_,_,k), tAsA);
    copy(tBgB(_,_,_,k), tBsB);

    __syncthreads();

    // Copy smem to rmem
    copy(smem_tiled_copy_A, tCsA, tCrA_copy_view);
    copy(smem_tiled_copy_B, tCsB, tCrB_copy_view);

    // Compute gemm on rmem
    gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
  }

  //
  // Epilogue
  //
  axpby(alpha, tCrC, beta, tCgC);
}


template <typename TA, typename TB, typename TC,
          typename Alpha, typename Beta>
void
gemm(int m, int n, int k,
     Alpha alpha,
     TA const* A, int ldA,
     TB const* B, int ldB,
     Beta beta,
     TC      * C, int ldC)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);

  // Define strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);

  // Define block sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 32>{};

  // Define the block layouts (static)
  auto sC = make_layout(make_shape(bM,bN));

  auto sA = tile_to_shape(SmemLayoutAtomA{}, make_shape(bM,bK));
  auto sB = tile_to_shape(SmemLayoutAtomB{}, make_shape(bN,bK));

  dim3 dimBlock(128, 1, 1);
  dim3 dimGrid(ceil_div(size(M), size(bM)),
               ceil_div(size(N), size(bN)));

  utils::launch_kernel<gemm_device<int, int, int,
          TA, decltype(dA),
          TB, decltype(dB),
          TC, decltype(dC),
          Alpha, Beta>>
          (dimGrid, dimBlock, 0, M,  N,  K,
           A, dA,
           B, dB,
           C, dC,
           alpha, beta);
}

int main(int argc, char** argv)
{
  int m = 5120;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 5120;
  if (argc >= 3)
    sscanf(argv[2], "%d", &n);

  int k = 4096;
  if (argc >= 4)
    sscanf(argv[3], "%d", &k);

  using TA = float;
  using TB = float;
  using TC = float;
  using TI = float;

  // This MMA operation casts the input matrices to half type, causing precision issues
  // when comparing with cublas output. Cast the input matrices to integers to avoid
  // these issues.
  using TAT = int;
  using TBT = int;

  utils::test_gemm<gemm<TA, TB, TC, TI, TI>, TA, TB, TC, TI, TAT, TBT>(m, n, k);

  return 0;
}
