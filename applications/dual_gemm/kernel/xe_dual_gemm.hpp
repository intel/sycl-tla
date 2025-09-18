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
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/tensor.hpp"

#include "dual_gemm/collective/xe_dual_gemm_mma.hpp"

namespace cutlass::gemm::kernel {

template <
  class ProblemShape_,
  class DualGemmMainloop_,
  class CollectiveEpilogue0_,
  class CollectiveEpilogue1_,
  class DualGemmElemActEpilogue_,
  class TileScheduler_ = void
>
class DualGemm;

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class DualGemmMainloop_,
  class CollectiveEpilogue0_,
  class CollectiveEpilogue1_,
  class DualGemmElemActEpilogue_,
  class TileScheduler_
>
class DualGemm
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using DualGemmMainloop = DualGemmMainloop_;
  using TileShape = typename DualGemmMainloop::WorkgroupTileShape;
  using WorkgroupTileShape = TileShape;
  using TiledMma  = typename DualGemmMainloop::TiledMma;
  using ArchTag   = typename DualGemmMainloop::ArchTag;
  using ElementA  = typename DualGemmMainloop::ElementA;
  using StrideA   = typename DualGemmMainloop::StrideA;
  using ElementB  = typename DualGemmMainloop::ElementB;
  using StrideB   = typename DualGemmMainloop::StrideB;
  using DispatchPolicy = typename DualGemmMainloop::DispatchPolicy;
  using ElementAccumulator = typename DualGemmMainloop::ElementAccumulator;
  using MainloopArguments = typename DualGemmMainloop::Arguments;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopParams = typename DualGemmMainloop::Params;

  static_assert(cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, PersistentScheduler>,
    "Intel Xe does not support specializing the tile scheduler.");
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileScheduler_, ArchTag, WorkgroupTileShape,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue0 = CollectiveEpilogue0_;
  using CollectiveEpilogue1 = CollectiveEpilogue1_;
  using ElementC = typename CollectiveEpilogue0::ElementC;
  using StrideC  = typename CollectiveEpilogue0::StrideC;
  using ElementD = typename CollectiveEpilogue0::ElementD;
  using StrideD  = typename CollectiveEpilogue0::StrideD;
  using EpilogueArguments0 = typename CollectiveEpilogue0::Arguments;
  using EpilogueArguments1 = typename CollectiveEpilogue1::Arguments;
  using EpilogueParams0 = typename CollectiveEpilogue0::Params;
  using EpilogueParams1 = typename CollectiveEpilogue1::Params;

  using DualGemmElemActEpilogue = DualGemmElemActEpilogue_;
  using DualGemmElemActEpilogueArguments = typename DualGemmElemActEpilogue::Arguments;
  using DualGemmElemActEpilogueParams = typename DualGemmElemActEpilogue::Params;
  static_assert(cute::is_same_v<ElementAccumulator, typename CollectiveEpilogue0::ElementAccumulator>,
    "Mainloop and epilogue do not agree on accumulator value type.");

  // MSVC requires the cast to fix a warning-as-error.
  static constexpr int SharedStorageSize = 0;

  static constexpr int SubgroupSize = DualGemmMainloop::SubgroupSize; // sub_group size
  static constexpr uint32_t MaxThreadsPerBlock = DualGemmMainloop::MaxThreadsPerBlock;
  using MmaAtomShape = typename DualGemmMainloop::MmaAtomShape;
  using SubgroupTileShape = typename DualGemmMainloop::SubgroupTileShape;
 
  using TensorMKL = typename DualGemmMainloop::TensorMKL;
  using TensorNKL = typename DualGemmMainloop::TensorNKL;
  using MainloopTensors = cute::tuple<TensorMKL, TensorNKL>;
  using TensorMK = decltype(TensorMKL{}(_, _, 0));
  using TensorNK = decltype(TensorNKL{}(_, _, 0));

  // Kernel level shared memory storage
  struct SharedStorage {
    using EpilogueTensorStorage0 = typename CollectiveEpilogue0::TensorStorage;
    using EpilogueTensorStorage1 = typename CollectiveEpilogue1::TensorStorage;
    EpilogueTensorStorage0 epilogue0;
    EpilogueTensorStorage1 epilogue1;
  };

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments0 epilogue0{};
    EpilogueArguments1 epilogue1{};
    DualGemmElemActEpilogueArguments elem_act_epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams0 epilogue0{};
    EpilogueParams1 epilogue1{};
    DualGemmElemActEpilogueParams elem_act_epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;
    auto problem_shape_MNKL = append<4>(args.problem_shape, 1);

    auto mainloop_args = DualGemmMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace);
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
      problem_shape_MNKL, TileShape{}, ClusterShape{}, args.hw_info, args.scheduler, &workspace);
    return {
      args.mode,
      args.problem_shape,
      mainloop_args,
      CollectiveEpilogue0::to_underlying_arguments(args.problem_shape, args.epilogue0, workspace),
      CollectiveEpilogue1::to_underlying_arguments(args.problem_shape, args.epilogue1, workspace),
      DualGemmElemActEpilogue::to_underlying_arguments(args.problem_shape, args.elem_act_epilogue, workspace),
      args.hw_info,
      scheduler
    };
  }

  static bool
  can_implement(Arguments const& args) {
    auto m = get<0>(args.problem_shape);
    auto n = get<1>(args.problem_shape);
    auto k = get<2>(args.problem_shape);
    // TODO(codeplay): base *_valid on the atom shapes
    bool m_valid = m > 0;
    bool n_valid = n > 0 && n % 4 == 0;
    bool k_valid = k > 0 && k % get<2>(TileShape{}) == 0;
    bool shape_implementable = (m_valid && n_valid && k_valid);

    bool mode_implementable = args.mode == GemmUniversalMode::kGemm ||
          (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
    return shape_implementable && mode_implementable && TileScheduler::can_implement(args.scheduler);
  }

  static int
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  static
  cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr, 
    CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3
  get_grid_shape(Params const& params) {
    dim3 grid = TileScheduler::get_tiled_cta_shape_mnl(params.problem_shape, TileShape{}, ClusterShape{});
    if(params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN) {
      return {grid.y, grid.x, grid.z};
    } else {
      return {grid.x, grid.y, grid.z};
    }
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    // Preconditions
    CUTE_STATIC_ASSERT(is_static<WorkgroupTileShape>::value);

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this sub_group -- potential for sub_group locality
    int thread_idx = int(ThreadIdxX());
    auto blk_shape = TileShape{};
    int m_coord, n_coord, l_coord;
    if (params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN) {
      m_coord = BlockIdxY();
      n_coord = BlockIdxX();
      l_coord = BlockIdxZ();
    } else {
      m_coord = BlockIdxX();
      n_coord = BlockIdxY();
      l_coord = BlockIdxZ();
    }

    auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);
    constexpr auto workgroup_shape = WorkgroupTileShape{};                                                  // (SUB_M,SUB_N,SUB_K)
    constexpr auto subgroup_shape = SubgroupTileShape{};                   

    Tensor mA_mkl = cute::get_xe_tensor(make_shape(M,K,L));   //(m,k,l)
    Tensor mB_nkl = cute::get_xe_tensor(make_shape(N,K,L));   //(n,k,l)

    Tensor gA = local_tile(mA_mkl, select<0,2>(blk_shape), make_coord(m_coord,_,l_coord));
    Tensor gB = local_tile(mB_nkl, select<1,2>(blk_shape), make_coord(n_coord,_,l_coord));

    // Compute tile residues for predication
    auto m_max_coord = M - get<0>(subgroup_shape) * m_coord;                             // M - SUB_M * m_coord
    auto n_max_coord = N - get<1>(subgroup_shape) * n_coord;                             // N - SUB_N * n_coord
    auto k_residue   = K - get<2>(subgroup_shape) * (K / get<2>(subgroup_shape));        // K - SUB_K * k_coord_max
    auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

    // Allocate the tiled_mma and the accumulators for the (M,N) subgroup_shape
    TiledMma tiled_mma;

    Tensor accumulators0 = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));
    Tensor accumulators1 = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));
    clear(accumulators0);
    clear(accumulators1);

    auto k_tile_iter  = cute::make_coord_iterator(idx2crd(0, make_shape(K)), make_shape(K));
    int  k_tile_count = K / get<2>(workgroup_shape);

    // Perform the collective scoped MMA
    DualGemmMainloop collective_mma;
    collective_mma(
      accumulators0,
      accumulators1,
      gA,
      gB,
      k_tile_iter, k_tile_count,
      residue_mnk,
      blk_coord_mnkl,
      K,
      thread_idx,
      smem_buf,
      params.mainloop
    );

    CollectiveEpilogue0 epilogue0{params.epilogue0, shared_storage.epilogue0};
    epilogue0(
      problem_shape_MNKL,
      subgroup_shape,
      blk_coord_mnkl,
      accumulators0,
      tiled_mma,
      residue_mnk,
      thread_idx,
      smem_buf
    );

    CollectiveEpilogue1 epilogue1{params.epilogue1, shared_storage.epilogue1};
    epilogue1(
      problem_shape_MNKL,
      subgroup_shape,
      blk_coord_mnkl,
      accumulators1,
      tiled_mma,
      residue_mnk,
      thread_idx,
      smem_buf
    );

    DualGemmElemActEpilogue elem_act_epilogue{params.elem_act_epilogue};
    elem_act_epilogue(
      problem_shape_MNKL,
      subgroup_shape,
      blk_coord_mnkl,
      accumulators0,
      accumulators1,
      tiled_mma,
      residue_mnk,
      thread_idx,
      smem_buf
    );
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
