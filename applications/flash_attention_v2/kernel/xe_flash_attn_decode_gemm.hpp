/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "flash_attention_v2/collective/xe_flash_decode_mma.hpp"

namespace cutlass::flash_attention::kernel {

template <class ProblemShape, class CollectiveMainloop, class CollectiveSoftmaxEpilogue_, class CollectiveEpilogue, class TileScheduler_ = void>
class FMHADecode;

///////////////////////////////////////////////////////////////////////////////

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveSoftmaxEpilogue_, class CollectiveEpilogue_, class TileScheduler_>
class FMHADecode {

public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  static_assert(rank(ProblemShape{}) == 8, "ProblemShape{} should be <batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, seq_len_kv_cache, head_size_qk, head_size_vo>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using WorkgroupTileShape = typename CollectiveMainloop::WorkgroupTileShape;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ElementQ = typename CollectiveMainloop::ElementQ;
  using StrideQ = typename CollectiveMainloop::StrideQ;
  using ElementK = typename CollectiveMainloop::ElementK;
  using StrideK = typename CollectiveMainloop::StrideK;
  using ElementV = typename CollectiveMainloop::ElementV;
  using StrideV = typename CollectiveMainloop::StrideV;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using CollectiveSoftmaxEpilogue = CollectiveSoftmaxEpilogue_;
  using SoftmaxArguments = typename CollectiveSoftmaxEpilogue::Arguments;
  using SoftmaxParams = typename CollectiveSoftmaxEpilogue::Params;

  static_assert(cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, FlashDecodeIndividualScheduler>,
                "Unsupported TileScheduler for Intel PVC.");
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler =
      typename detail::TileSchedulerSelector<TileScheduler_, ArchTag>::Scheduler;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementO = typename CollectiveEpilogue::ElementO;
  using StrideO = typename CollectiveEpilogue::StrideO;
  using ElementLSE = typename CollectiveEpilogue::ElementLSE;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static_assert(cute::is_same_v<ElementAccumulator, typename CollectiveEpilogue::ElementAccumulator>,
                "Mainloop and epilogue do not agree on accumulator value type.");

  static constexpr int SharedStorageSize = 0;

  static constexpr bool CausalMask = CollectiveMainloop::CausalMask;
  static constexpr int SubgroupSize = CollectiveMainloop::SubgroupSize; // sub_group size
  static constexpr uint32_t MaxThreadsPerBlock = CollectiveMainloop::MaxThreadsPerBlock;
  using MmaAtomShape = typename CollectiveMainloop::MmaAtomShape;           // 8,16,16

  static constexpr int QK_BLK_M = CollectiveMainloop::QK_BLK_M;
  static constexpr int QK_BLK_N = CollectiveMainloop::QK_BLK_N;
  static constexpr int QK_BLK_K = CollectiveMainloop::QK_BLK_K;

  static constexpr int QK_ATOM_N = CollectiveMainloop::QK_ATOM_N;
  static constexpr int QK_ATOM_K = CollectiveMainloop::QK_ATOM_K;

  using SubgroupTileShapeQK = typename CollectiveMainloop::SubgroupTileShapeQK;
  static constexpr int QK_SG_M = CollectiveMainloop::QK_SG_M;
  static constexpr int QK_SG_N = CollectiveMainloop::QK_SG_N;

  static constexpr int PV_BLK_N = CollectiveMainloop::PV_BLK_N;
  static constexpr int PV_BLK_K = CollectiveMainloop::PV_BLK_K;

  static constexpr int PV_ATOM_M = CollectiveMainloop::PV_ATOM_M;
  static constexpr int PV_ATOM_N = CollectiveMainloop::PV_ATOM_N;
  static constexpr int PV_ATOM_K = CollectiveMainloop::PV_ATOM_K;

  static constexpr auto Num_SGs = PV_ATOM_N * PV_ATOM_M * PV_ATOM_K;
  static constexpr int Vec = (get<0>(MmaAtomShape()) * get<1>(MmaAtomShape())) / SubgroupSize; // 8
  using FragsShapeQK = typename CollectiveMainloop::FragsShapeQK;
  using FragsShapePV = typename CollectiveMainloop::FragsShapePV;
  static constexpr int FragsM = get<0>(FragsShapeQK{});                                          // 1
  static constexpr int FragsN = get<1>(FragsShapeQK{});                                          // 4

  static_assert(FragsM == get<0>(FragsShapePV{}) == 1, "Limit the seq_len_qo to 1 MMA Atom worth of data per work-group.");

  static constexpr bool is_var_len = CollectiveMainloop::is_var_len;

  // Kernel level shared memory storage
  struct SharedStorage {
    using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
    EpilogueTensorStorage epilogue;
  };

  // Device side arguments
  struct Arguments {
    gemm::GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    SoftmaxArguments softmax{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
  };

  // Kernel entry point API
  struct Params {
    gemm::GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    SoftmaxParams softmax;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static Params to_underlying_arguments(Arguments const &args, void *workspace) {
    (void)workspace;
    return {args.mode, args.problem_shape,
            CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
            CollectiveSoftmaxEpilogue::to_underlying_arguments(args.softmax),
            CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
            TileScheduler::to_underlying_arguments(args.problem_shape, args.hw_info, WorkgroupTileShape{})};
  }

  static bool can_implement(Arguments const &args) {
    bool mode_implementable = args.mode == gemm::GemmUniversalMode::kGemm or
                              (args.mode == gemm::GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
    return mode_implementable;
  }

  static int get_workspace_size(Arguments const &args) { return 0; }

  static cutlass::Status initialize_workspace(Arguments const &args, void *workspace = nullptr,
                                              cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const &params) {
    return TileScheduler::template get_grid_shape<Num_SGs>(params.scheduler);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE
  Shape<int, int, int, int, int, int, int, int> get_logical_problem_shape(ProblemShape const& problem_shape, int const& batch) {
    if constexpr (is_var_len) {
      return cutlass::fmha::collective::apply_variable_length(problem_shape, batch);
    } else {
      return problem_shape;
    }
  }

  CUTLASS_DEVICE
  void operator()(Params const &params, char *smem_buf) {
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);
    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShapeQK>::value);
    CUTE_STATIC_ASSERT(is_static<TileShapePV>::value);
    // Separate out problem shape for convenience
    auto& batch = get<0>(params.problem_shape);
    auto& num_heads_q = get<1>(params.problem_shape);
    auto& num_heads_kv = get<2>(params.problem_shape);
    auto group_heads_q = num_heads_q / num_heads_kv;
    auto& head_size_qk = get<6>(params.problem_shape);
    auto& head_size_vo = get<7>(params.problem_shape);
    // Preconditions
    static_assert(cute::rank(StrideQ{}) == 3, "StrideQ must be rank-3: [seq_len_qo, head_size_qk, batch * num_heads_q].");
    static_assert(cute::rank(StrideK{}) == 3, "StrideK must be rank-3: [head_size_qk, seq_len_kv, batch * num_heads_kv].");
    static_assert(cute::rank(StrideV{}) == 3, "StrideV must be rank-3: [seq_len_kv, head_size_vo, batch * num_heads_kv].");

    int thread_idx = int(ThreadIdxX());
    int sub_group_id = thread_idx / SubgroupSize;

    TileScheduler tile_scheduler{params.scheduler};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto blk_coord = tile_scheduler.get_block_coord(); // head_size_blk_idx, seq_len_blk_idx, batch_blk_idx, num_heads_blk_idx

      auto blk_q_coord = get<1>(blk_coord); // seq_len_q_blk_idx
      auto blk_v_coord = get<0>(blk_coord); // head_size_blk_idx
      auto batch_coord = get<2>(blk_coord); // batch_blk_idx
      auto num_heads_coord = get<3>(blk_coord); // num_heads_blk_idx

      // For variable sequence length case, batch is considered to be 1 (same as group gemm).
      // For fixed sequence length case, the l_coord is the weighted sum of both batch_coord and num_heads_coord.
      // Flash Attention implementation combines batch and num_heads to calculate the total batch_size.
      // iff is_var_len: batch_size = num_heads (as each batch would have it's own seq_len_qo and seq_len_kv)
      // iff !is_var_len: batch_size = batch * num_heads
      auto blk_l_coord = is_var_len ? num_heads_coord : batch_coord * num_heads_q + num_heads_coord;

      // Get problem shape for the current batch_blk_idx. For variable sequence length, it loads the sequence length
      // from Global memory for the given batch_blk_idx and returns the appropriate problem_shape. For fixed sequence
      // length, logical_problem_shape == params.problem_shape.
      // logical_problem_shape = [batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, seq_len_kv_cache, head_size_qk, head_size_vo]
      auto logical_problem_shape = get_logical_problem_shape(params.problem_shape, batch_coord);

      auto [seq_len_qo, seq_len_kv, seq_len_kv_cache] = select<3, 4, 5>(logical_problem_shape);

      Tensor mQ_mkl = cute::get_pvc_tensor(make_shape(seq_len_qo, head_size_qk, (is_var_len ? 1 : batch) * num_heads_q));   //(m,k,l)
      Tensor mK_nkl = cute::get_pvc_tensor(make_shape(seq_len_kv, head_size_qk, (is_var_len ? 1 : batch) * num_heads_kv));   //(n,k,l)
      Tensor mV_nkl = cute::get_pvc_tensor(make_shape(head_size_vo, seq_len_kv, (is_var_len ? 1 : batch) * num_heads_kv));   //(n,k,l)
      Tensor mK_cache_nkl = cute::get_pvc_tensor(make_shape(seq_len_kv_cache, head_size_qk, (is_var_len ? 1 : batch) * num_heads_kv));   // (n_cache,k,l)
      Tensor mV_cache_nkl = cute::get_pvc_tensor(make_shape(head_size_vo, seq_len_kv_cache, (is_var_len ? 1 : batch) * num_heads_kv));   // (n_cache,k,l)

      Tensor mQ_mk = mQ_mkl(_, _, blk_l_coord);                                                    // (m,k)
      Tensor mK_nk = mK_nkl(_, _, blk_l_coord / group_heads_q);                                                    // (n,k)
      Tensor mV_nk = mV_nkl(_, _, blk_l_coord / group_heads_q);                                                    // (n,k)
      Tensor mK_cache_nk = mK_cache_nkl(_, _, blk_l_coord / group_heads_q);                                        // (n_cache, k)
      Tensor mV_cache_nk = mV_cache_nkl(_, _, blk_l_coord / group_heads_q);                                        // (n_cache, k)

      auto gQ = local_tile(mQ_mk, TileShapeQK{}, make_coord(blk_q_coord, _, _), Step<_1,  X, _1>{});
      auto gK_prefetch = local_tile((seq_len_kv_cache == 0) ? mK_nk : mK_cache_nk, SubgroupTileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
      auto gV = local_tile((seq_len_kv_cache == 0) ? mV_nk : mV_cache_nk, TileShapePV{}, make_coord(_, blk_v_coord, _), Step<X, _1, _1>{});

      // Determine how many tiles are supposed to be processed using this subgroup
      const int kv_splits_new = ceil_div(seq_len_kv, get<1>(TileShapeQK{}));
      const int kv_splits_cache = ceil_div(seq_len_kv_cache, get<1>(TileShapeQK{}));
      const int kv_splits = kv_splits_new + kv_splits_cache;

      auto mainloop_params = CollectiveMainloop::get_updated_copies(params.mainloop, params.problem_shape, batch_coord);
      // TODO(Codeplay): the 32 is the maximum row we can commit. This block need to get fixed via passing correct block number to read
      static constexpr auto QK_BLK_N_prefetch =  32 * 4 * PV_ATOM_N;   
      auto tiled_prefetch_q = cute::prefetch_selector<Shape<Int<QK_BLK_M>, Int<QK_BLK_K>>, Num_SGs>(mainloop_params.gmem_tiled_copy_q);
      auto tiled_prefetch_k = cute::prefetch_selector<Shape<Int<QK_BLK_N_prefetch>, Int<QK_BLK_K>>, Num_SGs>((seq_len_kv_cache == 0) ? 
                                                                                                              mainloop_params.gmem_tiled_copy_k :
                                                                                                              mainloop_params.gmem_tiled_copy_k_cache);
      auto tiled_prefetch_v = cute::prefetch_selector<Shape<Int<PV_BLK_N>, Int<PV_BLK_K>>, Num_SGs>((seq_len_kv_cache == 0) ?
                                                                                                      mainloop_params.gmem_tiled_copy_v :
                                                                                                      mainloop_params.gmem_tiled_copy_v_cache);
      auto thr_prefetch_Q = tiled_prefetch_q.get_slice(thread_idx);
      auto thr_prefetch_K = tiled_prefetch_k.get_slice(thread_idx);
      auto thr_prefetch_V = tiled_prefetch_v.get_slice(thread_idx);
      auto pQgQ = thr_prefetch_Q.partition_S(gQ);
      auto pKgK = thr_prefetch_K.partition_S(gK_prefetch);
      auto pVgV = thr_prefetch_V.partition_S(gV);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<3>(pQgQ); i++) {
        prefetch(tiled_prefetch_q, pQgQ(_, _, _, i));
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < QK_BLK_N / QK_BLK_N_prefetch ; i++) {
        // The headsize for both cached and non-cached version is the same
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size<4>(pKgK); j++) {
          prefetch(tiled_prefetch_k, pKgK(_, _, _, i, j));
        }
      }

      // Allocate the tiled_mma and the accumulators for the (M,N) workgroup_shape
      TiledMma tiled_mma;
      // Perform the collective scoped MMA
      CollectiveMainloop collective_mma;
      // Barrier scope should always be set to subgroup since we use SplitK
      // algorithm to distribute the K and V matrices across multiple subgroups.
      // Setting the scope to workgroup leads to race condition.
      static constexpr int barrier_scope = 3;
      int kv_tile_idx = sub_group_id / PV_ATOM_N;

      ElementAccumulator max_reg = ElementAccumulator{-INFINITY};
      Tensor sum_reg = make_tensor<ElementAccumulator>(Shape<Int<Vec>, Int<FragsM>>{});
      Tensor out_reg = make_tensor<ElementAccumulator>(Shape<Int<Vec>, Int<get<0>(FragsShapePV{})>, Int<get<1>(FragsShapePV{})>>{});
      clear(out_reg);
      clear(sum_reg);

      auto smem = syclcompat::local_mem<ElementAccumulator[(((Vec * get<0>(FragsShapePV{}) * get<1>(FragsShapePV{})) + 1) * Num_SGs * SubgroupSize)]>();

      if constexpr (!CausalMask) {
        CUTLASS_PRAGMA_NO_UNROLL
        for(int split = 0; split < kv_splits; split++) {
          barrier_arrive(barrier_scope);

          Tensor shmem_max_tensor = make_tensor(make_smem_ptr(smem), make_shape(Int<Num_SGs * Vec * FragsM>{}));
          
          bool is_KV_cache = split < kv_splits_cache;
          auto gK = is_KV_cache ? local_tile(mK_cache_nk, TileShapeQK{}, make_coord(_, split, _), Step<X, _1, _1>{})
                                : local_tile(mK_nk, TileShapeQK{}, make_coord(_, split - kv_splits_cache, _), Step<X, _1, _1>{});

          // 1) Load K (performed inside mmaQK)
          // 2) Create Tensor S
          Tensor tSr = make_tensor<ElementAccumulator>(Shape<Int<Vec>, Int<FragsM>, Int<FragsN>>{});
          clear(tSr);

          // 3) Perform GEMM S = Q*K
          collective_mma.mmaQK(tSr, gQ, gK, tSr, ceil_div(head_size_qk, QK_BLK_K), mainloop_params, is_KV_cache);

          // we only need one block ahead, there is enough gap to prefetch it while doing softmax. because the gap between the two MMA is big,
          // prefetching it the same way as cutlass K matrix does not make sense
          prefetch(tiled_prefetch_v, pVgV(_, _, _, kv_tile_idx + (split - (is_KV_cache ? 0 : kv_splits_cache)) * PV_ATOM_M));

          CollectiveSoftmaxEpilogue softmax(params.softmax);
          softmax.template operator()<Num_SGs>(split == 0, tSr, max_reg, sum_reg, shmem_max_tensor, out_reg);

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i <  QK_BLK_N / QK_BLK_N_prefetch; i++) {
            // The headsize for both cached and non-cached version is the same
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size<4>(pKgK); j++) {
              prefetch(tiled_prefetch_k, pKgK(_, _, _, i + (split + 1 - (is_KV_cache ? 0 : kv_splits_cache)) * (QK_BLK_N / QK_BLK_K), j));
            }
          }

          auto gV_ = gV(_, _, kv_tile_idx + (split - (is_KV_cache ? 0 : kv_splits_cache)) * PV_ATOM_M);

          collective_mma.mmaPV(out_reg, tSr, gV_, out_reg, mainloop_params, is_KV_cache);

          // Prefetch the next Q tile
          // there is no need to guard it with if statememt as prefetch will ignore out of bound reading
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size<3>(pQgQ); i++) {
            prefetch(tiled_prefetch_q, pQgQ(_, _, _, i));
          }

          barrier_wait(barrier_scope);
        }
      } else {

        CUTLASS_PRAGMA_NO_UNROLL
        for(int split = 0; split < kv_splits; split++) {
          barrier_arrive(barrier_scope);

          // 1) Load K (performed inside mmaQK)
          // 2) Create Tensor S
          Tensor tSr = make_tensor<ElementAccumulator>(Shape<Int<Vec>, Int<FragsM>, Int<FragsN>>{});
          clear(tSr);

          bool is_KV_cache = split < kv_splits_cache;
          auto gK = is_KV_cache ? local_tile(mK_cache_nk, TileShapeQK{}, make_coord(_, split, _), Step<X, _1, _1>{})
                                : local_tile(mK_nk, TileShapeQK{}, make_coord(_, split - kv_splits_cache, _), Step<X, _1, _1>{});
          // 3) Perform GEMM S = Q*K
          collective_mma.mmaQK(tSr, gQ, gK, tSr, ceil_div(head_size_qk, QK_BLK_K), mainloop_params, is_KV_cache);

          if(kv_tile_idx == PV_ATOM_M - 1 && split == kv_splits - 1) {
            int column_offset = seq_len_kv - seq_len_qo + seq_len_kv_cache;
            int col_idx = (kv_tile_idx + split * PV_ATOM_M) * QK_SG_N + thread_idx % SubgroupSize;
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < FragsN; n++, col_idx += get<1>(MmaAtomShape())) { // 4
              CUTLASS_PRAGMA_UNROLL
              for (int m = 0; m < FragsM; m++) { // 2
                int row_idx = m * Vec + blk_q_coord * QK_SG_M; // Use Vec based on seq_len_qo
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < Vec; row++, row_idx++) { // Set this bound based on seq_len_qo
                  if (col_idx - column_offset > row_idx)
                    tSr(row, m, n) = -INFINITY;
                }
              }
            }
          }

          // we only need one block ahead, there is enough gap to prefetch it while doing softmax. because the gap between the two MMA is big,
          // prefetching it the same way as cutlass K matrix does not make sense
          prefetch(tiled_prefetch_v, pVgV(_, _, _, kv_tile_idx + (split - (is_KV_cache ? 0 : kv_splits_cache)) * PV_ATOM_M));

          // Softmax needs to happen for invalid subgroups as well
          // because it has a barrier. Skipping Softmax computation
          // for some subgroups would result in a stall.
          Tensor shmem_max_tensor = make_tensor(make_smem_ptr(smem), make_shape(Int<Num_SGs * Vec * FragsM>{}));
          CollectiveSoftmaxEpilogue softmax(params.softmax);
          softmax.template operator()<Num_SGs>(split == 0, tSr, max_reg, sum_reg, shmem_max_tensor, out_reg);

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i <  QK_BLK_N / QK_BLK_N_prefetch; i++) {
            // The headsize for both cached and non-cached version is the same
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size<4>(pKgK); j++) {
              prefetch(tiled_prefetch_k, pKgK(_, _, _, i + (split + 1 - (is_KV_cache ? 0 : kv_splits_cache)) * (QK_BLK_N / QK_BLK_K), j));
            }
          }

          auto gV_ = gV(_, _, kv_tile_idx + (split - (is_KV_cache ? 0 : kv_splits_cache)) * PV_ATOM_M);

          collective_mma.mmaPV(out_reg, tSr, gV_, out_reg, mainloop_params, is_KV_cache);
          // Prefetch the next Q tile
          // there is no need to guard it with if statememt as prefetch will ignore out of bound reading
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size<3>(pQgQ); i++) {
            prefetch(tiled_prefetch_q, pQgQ(_, _, _, i));
          }

          barrier_wait(barrier_scope);
        }
      }

      Tensor shmem_out_tensor = make_tensor(make_smem_ptr(smem), make_shape(Int<(Vec * get<0>(FragsShapePV{}) * get<1>(FragsShapePV{})) * SubgroupSize * Num_SGs>{}));
      // write output to SLM
      int idx = (thread_idx % SubgroupSize) + sub_group_id * out_reg.size() * SubgroupSize;
      CUTLASS_PRAGMA_UNROLL
      for(int i = 0; i < Vec * get<0>(FragsShapePV{}) * get<1>(FragsShapePV{}); i++) {
        shmem_out_tensor(idx + i * SubgroupSize) = out_reg(i);
      }

      auto epilogue_params = CollectiveEpilogue::template get_updated_copies<is_var_len>(params.epilogue, params.problem_shape, batch_coord);
      CollectiveEpilogue epilogue{epilogue_params, shared_storage.epilogue};
      auto blk_coord_mnkl = make_coord(blk_q_coord, blk_v_coord, _, blk_l_coord);

      Tensor shmem_sum_tensor = make_tensor(make_smem_ptr(shmem_out_tensor.data() + shmem_out_tensor.size()), make_shape(Int<Num_SGs * Vec * FragsM>{}));

      epilogue(logical_problem_shape, TileShapePV{}, blk_coord_mnkl, shmem_out_tensor, sum_reg, shmem_sum_tensor, tiled_mma);
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::flash_attention::kernel
