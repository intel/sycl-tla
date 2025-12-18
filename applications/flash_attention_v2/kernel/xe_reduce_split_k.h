/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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
/*! \file
  \brief Kernel performing a reduction over densely packed tensors in global memory
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp"
#include "cute/util/type_traits.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reduction {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class ProblemShape_,
          class TileScheduler_,
          class FMHAKernel_
>
class ReduceSplitK {
public:

  using ProblemShape = ProblemShape_;
  using VariableLength = cutlass::fmha::collective::VariableLength;
  static constexpr bool is_var_len = cutlass::fmha::collective::is_variable_length_v<typename ProblemShape::SeqLenType>;
  using TileScheduler = TileScheduler_;
  static_assert(is_same_v<TileScheduler, cutlass::fmha::kernel::XeReduceSplitKTileScheduler>,
                "ReduceSplitK kernel requires XeReduceSplitKTileScheduler");
  using TileSchedulerParams = typename TileScheduler::Params;

  using ElementO = typename FMHAKernel_::ElementO;
  using StrideO = typename FMHAKernel_::StrideO;
  using TileShapeO = typename FMHAKernel_::TileShapeO;
  using TileShapeQK = typename FMHAKernel_::TileShapeQK;

  using SGPerWG = typename FMHAKernel_::SGPerWG;

  // num values (head_dim) processed by each thread
  constexpr static int num_vals_per_thread = int(get<1>(TileShapeO{}) / (SGPerWG::value * intel::sg_size));

  //
  // Types
  //

  struct KernelArguments {
    ProblemShape shape;
    // outputs:
    ElementO *O;
    StrideO dO;
    // below are inputs
    // TODO: whether same dtype as output or accum?
    const ElementO *Oaccum;
    StrideO dOaccum;
    const ElementO *exp_sums;
    StrideO dExp_sums;
    const ElementO *max_logits;
    StrideO dMax_logits;
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    KernelHardwareInfo hw_info{};
    int num_kv_splits = -1; // no split by default
  };

  /// Params structure
  struct Params {
    KernelParams kernel;
    TileSchedulerParams scheduler;
  };

  struct SharedStorage {
    cutlass::Array<ElementO, FMHAKernel_::max_num_kv_splits> max_logits_slm_array;
    cutlass::Array<ElementO, FMHAKernel_::max_num_kv_splits> exp_sums_slm_array;
  };

  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0)
                                                                     : sizeof(SharedStorage);

public:

  static Params to_underlying_arguments(Arguments const &args, void *workspace) {
    return {args.kernel,
            TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{}, args.num_kv_splits)};
  }

  static bool can_implement(Arguments const &args) {
    // only support decode
    if (!is_var_len && args.kernel.shape.seq_len_qo > 1) {
      return false;
    }

    if (args.num_kv_splits > FMHAKernel_::max_num_kv_splits) {
      return false;
    }
    return true;
  }

  static int get_workspace_size(Arguments const &args) { return 0; }

  static cutlass::Status initialize_workspace(Arguments const &args, void *workspace = nullptr,
                                              cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const &params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() { return dim3(SGPerWG::value * intel::sg_size, 1, 1); }

  CUTLASS_DEVICE
  Shape<int, int> get_sequence_length_shape(ProblemShape const& problem_shape, int const& batch) {
    if constexpr (is_var_len) {
      return cutlass::fmha::collective::apply_variable_length(Shape<VariableLength, VariableLength>{problem_shape.seq_len_qo, problem_shape.seq_len_kv}, batch);
    } else {
      return Shape<int, int>{problem_shape.seq_len_qo, problem_shape.seq_len_kv};
    }
  }

  /// Perform a reduction
  CUTLASS_DEVICE
  void operator()(Params const &params, char *smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    auto &p = params.kernel;
    ProblemShape const& s = p.shape;

    int thr_id = int(ThreadIdxX());
    int sub_group_id = thr_id / intel::sg_size;
    int tid_in_sg = thr_id % intel::sg_size;

    TileScheduler tile_scheduler{params.scheduler};
    auto num_kv_splits = params.scheduler.num_kv_splits;

    auto batch_dim = is_var_len ? 1 : s.batch;
    auto num_heads_q = s.num_heads_q;
    auto head_size_vo = s.head_size_vo;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [seq_idx, head_q, idx_b] = tile_scheduler.get_block_coord();

      auto sequence_length_shape = get_sequence_length_shape(s, idx_b);
      auto [seq_len_qo, seq_len_kv] = sequence_length_shape;

      // when varlen enabled, use largest seq_len_qo to decide work group num
      if (seq_idx >= seq_len_qo) continue;

      const int k_blocks = cute::ceil_div(seq_len_kv, get<1>(TileShapeQK{}));
      int num_blocks_per_split = cute::ceil_div(k_blocks, num_kv_splits);

      int offset_o = 0, offset_o_accum = 0;
      int offset_exp_sums = 0, offset_max_logits = 0;

      if constexpr (is_var_len) {
        auto qo_cumulative = s.seq_len_qo.cumulative_length;

        offset_o_accum = s.num_heads_q * s.head_size_vo * num_kv_splits * qo_cumulative[idx_b];
        offset_exp_sums = s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];
        offset_max_logits = s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];

        offset_o = s.num_heads_q * s.head_size_vo * qo_cumulative[idx_b];
      }

      auto shape_O = make_shape(seq_len_qo, head_size_vo, num_heads_q,  batch_dim);
      auto shape_Oaccum = is_var_len ? make_shape(seq_len_qo, head_size_vo, num_heads_q * num_kv_splits, batch_dim) : make_shape(seq_len_qo, head_size_vo, num_heads_q * num_kv_splits, batch_dim);

      auto shape_exp_sums = make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch_dim);
      auto shape_max_logits = make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch_dim);

      auto dcOaccum = const_cast<ElementO*>(p.Oaccum + offset_o_accum);
      auto ptrExp_sums = const_cast<ElementO*>(p.exp_sums + offset_exp_sums);
      auto ptrMax_logits = const_cast<ElementO*>(p.max_logits + offset_max_logits);
      auto ptrO = p.O + offset_o;

      auto stride_o = is_var_len ? cutlass::make_cute_packed_stride(StrideO{}, shape_O) : p.dO;
      auto stride_o_accum = is_var_len ? cutlass::make_cute_packed_stride(StrideO{}, shape_Oaccum) : p.dOaccum;
      auto stride_exp_sums = is_var_len ? cutlass::make_cute_packed_stride(StrideO{}, shape_exp_sums) : p.dExp_sums;
      auto stride_max_logits = is_var_len ? cutlass::make_cute_packed_stride(StrideO{}, shape_max_logits) : p.dMax_logits;

      Tensor Oaccum = make_tensor(make_gmem_ptr(dcOaccum), make_layout(shape_Oaccum, stride_o_accum));
      Tensor O = make_tensor(make_gmem_ptr(ptrO), make_layout(shape_O, stride_o));

      Tensor exp_sums = make_tensor(make_gmem_ptr(ptrExp_sums), make_layout(shape_exp_sums, stride_exp_sums));
      Tensor max_logits = make_tensor(make_gmem_ptr(ptrMax_logits), make_layout(shape_max_logits, stride_max_logits));

      int l_coord = is_var_len ? 0 : idx_b;

      // Step 1: reduce max logits across different partitions
      // store into SLM for later use

      ElementO global_max_logits = cutlass::platform::numeric_limits<ElementO>::lowest();
      ElementO global_exp_sums = 0;
      // only first subgroup participates
      if (thr_id < num_kv_splits) {
        ElementO cur_max_logit = max_logits(seq_idx, thr_id, head_q, l_coord);
        global_max_logits = sycl::max(global_max_logits, cur_max_logit);
        shared_storage.max_logits_slm_array[thr_id] = cur_max_logit;

        ElementO cur_exp_sum = exp_sums(seq_idx, thr_id, head_q, l_coord);
        shared_storage.exp_sums_slm_array[thr_id] = cur_exp_sum;
      }

      // barrier for SLM writes finished
      sycl::group_barrier(get_work_group<3>());

      // reduce across wg
      global_max_logits = reduce_over_group(get_work_group<1>(), global_max_logits, sycl::maximum<>());

      // broadcast to all other threads
      global_max_logits = sycl::group_broadcast(get_work_group<1>(), global_max_logits, 0);

      // step 2: rescale Oaccum and write back to O
      for (int idx = thr_id; idx < s.head_size_vo; idx += SGPerWG::value * intel::sg_size) {
        ElementO acc = 0;
        for (int i = 0; i < num_kv_splits; ++i) {
          if (i * num_blocks_per_split > k_blocks) {
            break;
          }
          ElementO local_max_logit = shared_storage.max_logits_slm_array[i];
          ElementO local_exp_sum = shared_storage.exp_sums_slm_array[i];

          ElementO rescale = sycl::native::exp2(local_max_logit - global_max_logits);

          // in FMHA epilogue, it's divided by local_exp_sum
          // assume seq_len_q == 1
          ElementO adjusted_o_accum = Oaccum(seq_idx, idx, i * num_heads_q + head_q, l_coord) * local_exp_sum;
          acc += adjusted_o_accum * rescale;

          // update global exp sum
          global_exp_sums += local_exp_sum * rescale;
        }

        ElementO inv_global_exp_sums = 1. / global_exp_sums;
        acc *= inv_global_exp_sums;
        O(seq_idx, idx, head_q, l_coord) = acc;
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace reduction
} // namespace cutlass
