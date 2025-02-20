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

/*! \file
    \brief Parameters structure for persistent tile schedulers
*/

#include "cutlass/coord.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/workspace.h"
#include "cutlass/platform/platform.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm_coord.h"
////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {
namespace detail {

// Parameters for Xe persistent group scheduler (only used for Grouped Gemms)
template<class ProblemShape>
struct PersistentTileSchedulerXeGroupParams {

  enum class RasterOrder {
    AlongM,
    AlongN
  };

  enum class RasterOrderOptions {
    Heuristic,
    AlongM,
    AlongN
  };

  FastDivmodU64Pow2 divmod_cluster_shape_major_{};
  FastDivmodU64Pow2 divmod_cluster_shape_minor_{};
  FastDivmodU64 divmod_wg_shape_m_{};
  FastDivmodU64 divmod_wg_shape_n_{};

  uint64_t blocks_across_problem_ = 0;
  bool pre_processed_problem_shapes = true;
  RasterOrder raster_order_ = RasterOrder::AlongN;

  int32_t groups_ = 0;
  ProblemShape* problem_shapes_ = nullptr;
  GemmCoord wg_shape_;
  GemmCoord cluster_shape_;

  // Version of initialize that takes in as input the number of WGs in the M and N and L dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or WG shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  void
  initialize(
    dim3 problem_blocks,
    int32_t groups,
    ProblemShape* problem_shapes,
    ProblemShape const* host_problem_shapes,
    GemmCoord wg_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    RasterOrderOptions raster_order_option
  ) {

    CUTLASS_UNUSED(hw_info);

    auto problem_blocks_m = problem_blocks.x;
    auto problem_blocks_n = problem_blocks.y;

    RasterOrder raster_order = get_rasterization_order(
      problem_blocks_m,
      problem_blocks_n,
      raster_order_option
    );

    //
    // Set members
    //
    groups_ = groups;
    problem_shapes_ = problem_shapes;
    wg_shape_ = wg_shape;
    cluster_shape_ = cluster_shape;

    blocks_across_problem_ = problem_blocks.x * problem_blocks.y * problem_blocks.z;
    pre_processed_problem_shapes = (host_problem_shapes == nullptr) ? false : true;
    raster_order_ = raster_order;

    if (raster_order == RasterOrder::AlongN) {
      divmod_cluster_shape_major_ = FastDivmodU64Pow2(cluster_shape.n());
      divmod_cluster_shape_minor_ = FastDivmodU64Pow2(cluster_shape.m());
    }
    else {
      divmod_cluster_shape_major_ = FastDivmodU64Pow2(cluster_shape.m());
      divmod_cluster_shape_minor_ = FastDivmodU64Pow2(cluster_shape.n());
    }

    divmod_wg_shape_m_ = FastDivmodU64(wg_shape_.m());
    divmod_wg_shape_n_ = FastDivmodU64(wg_shape_.n());
  }

  // Version of get_tiled_wg_shape_mnl that takes in as input the number of WGs in the M and N dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or WG shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  CUTLASS_HOST_DEVICE
  static dim3
  get_tiled_wg_shape_mnl(GemmCoord, uint32_t wg_m, uint32_t wg_n) {
    return {
      static_cast<uint32_t>(wg_m),
      static_cast<uint32_t>(wg_n),
      static_cast<uint32_t>(1) // Only a single batch per group is currently supported
    };
  }

  // Version of get_grid_shape that takes in as input the number of WGs in the M and N and L dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or WG shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  CUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
    dim3 problem_blocks,
    GemmCoord,
    KernelHardwareInfo hw_info,
    RasterOrderOptions raster_order_option,
    bool truncate_by_problem_size=true) {

    int const sm_count = hw_info.sm_count;

    auto problem_blocks_m = problem_blocks.x;
    auto problem_blocks_n = problem_blocks.y;

    int problem_blocks_total = problem_blocks_m * problem_blocks_n * problem_blocks.z;

    RasterOrder raster_order = get_rasterization_order(
      problem_blocks_m,
      problem_blocks_n,
      raster_order_option
    );

    dim3 launch_grid;

    auto possibly_truncate = [&](int x, int y) {
      if (truncate_by_problem_size) {
        return platform::min(x, y);
      }
      else {
        return x;
      }
    };

    if (raster_order == RasterOrder::AlongN)
    {
        launch_grid.y = possibly_truncate(sm_count / 8, problem_blocks_total); // divide EUs by 8 to get available_xe_cores
    }
    else
    {
        launch_grid.x = possibly_truncate(sm_count / 8, problem_blocks_total); // divide EUs by 8 to get available_xe_cores
    }

    return launch_grid;
  }

  CUTLASS_HOST_DEVICE
  static RasterOrder
  get_rasterization_order(
    uint32_t tiles_m,
    uint32_t tiles_n,
    RasterOrderOptions raster_order_option
  ) {

    if (raster_order_option == RasterOrderOptions::Heuristic) {
      if (tiles_n > tiles_m) {
        return RasterOrder::AlongM;
      }
      else {
        return RasterOrder::AlongN;
      }
    }
    else {
      switch (raster_order_option) {
        case RasterOrderOptions::AlongN:
          return RasterOrder::AlongN;
          break;
        default:
          return RasterOrder::AlongM;
      }
    }
  }
};

} // namespace detail
} // namespace kernel
} // namespace gemm
} // namespace cutlass
