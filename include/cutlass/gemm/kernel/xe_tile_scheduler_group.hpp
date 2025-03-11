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

#include "cutlass/fast_math.h"
#include "cutlass/gemm_coord.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp"

namespace cutlass::gemm::kernel::detail {

///////////////////////////////////////////////////////////////////////////////

// Persistent Thread Block (TB) scheduler
template <class GroupProblemShape>
class PersistentTileSchedulerXeGroup : public PersistentTileSchedulerSm90Group<GroupProblemShape> {

  using BaseScheduler = PersistentTileSchedulerSm90Group<GroupProblemShape>;
public:
  using Params = typename BaseScheduler::Params;
  using RasterOrder = typename BaseScheduler::RasterOrder;
  using Arguments = typename BaseScheduler::Arguments;

  CUTLASS_DEVICE explicit PersistentTileSchedulerXeGroup(Params const& params_) : BaseScheduler(params_) {}

  // Given the inputs, computes the physical grid we should launch.
  template<class TileShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
    [[maybe_unused]] Params const& params,
    GroupProblemShape problem_shapes,
    TileShape tile_shape,
    ClusterShape cluster_shape,
    KernelHardwareInfo hw_info,
    Arguments arguments,
    bool truncate_by_problem_size=true) {

    dim3 problem_blocks = BaseScheduler::get_tiled_cta_shape_mnl(
      problem_shapes.groups(),
      problem_shapes,
      hw_info,
      tile_shape, cluster_shape);

    int const sm_count = hw_info.sm_count;

    auto problem_blocks_m = problem_blocks.x;
    auto problem_blocks_n = problem_blocks.y;

    int problem_blocks_total = problem_blocks_m * problem_blocks_n * problem_blocks.z;

    RasterOrder raster_order = Params::get_rasterization_order(
      problem_blocks_m,
      problem_blocks_n,
      arguments.raster_order
    );

    dim3 launch_grid;

    auto possibly_truncate = [&](int x, int y) {
      return platform::min(x, y);
    };

    // divide EUs by 8 to get available_xe_cores for PVC
    if (raster_order == RasterOrder::AlongN) {
        launch_grid.y = possibly_truncate(sm_count / 8, problem_blocks_total);
    } else {
        launch_grid.x = possibly_truncate(sm_count / 8, problem_blocks_total);
    }

    return launch_grid;
  }
};

} // namespace cutlass::gemm::kernel::detail
