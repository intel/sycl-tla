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

#include "flash_attention_v2/collective/xe_flash_attn_prefill_mma.hpp"

namespace cutlass::flash_attention::kernel
{

  template <class ProblemShape_, class CollectiveMainloop_, class CollectiveSoftmaxEpilogue_, class CollectiveEpilogue_, class TileScheduler_ = void>
  class FMHAPrefillChunk;
///////////////////////////////////////////////////////////////////////////////
  template <class ProblemShape_, class CollectiveMainloop_, class CollectiveSoftmaxEpilogue_,
            class CollectiveEpilogue_, class TileScheduler_ = void>
  class FMHAPrefillChunk
  {

  public:
    //
    // Type Aliases
    //
    using ProblemShape = ProblemShape_;

    // ProblemShape: <batch, num_heads_q, num_head_kv, seq_len_qo, seq_len_kv, chunk_size, head_size_qk, head_size_vo>
    static_assert(rank(ProblemShape{}) == 8,
                  "ProblemShape{} should be <batch, num_heads_q, num_head_kv, seq_len_qo, seq_len_kv, chunk_size, head_size_qk, head_size_vo>");

    CUTLASS_DEVICE
    void operator()(Params const &params, char *smem_buf)
    {
      SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);
      // Preconditions
      CUTE_STATIC_ASSERT(is_static<TileShapeQK>::value);
      CUTE_STATIC_ASSERT(is_static<TileShapePV>::value);
      // Separate out problem shape for convenience

      // "ProblemShape{} should be <batch, num_heads_q, num_head_kv, seq_len_qo, seq_len_kv, chunk_size, head_size_qk, head_size_vo>");
      auto &batch = get<0>(params.problem_shape);
      auto &num_heads_q = get<1>(params.problem_shape);
      auto &num_head_kv = get<2>(params.problem_shape);
      auto group_heads_q = num_heads_q / num_head_kv;

      auto &seq_len_kv = get<4>(params.problem_shape); // 总的 seq_len_kv
      auto &chunk_size = get<5>(params.problem_shape); // 获取 chunk_size

      auto &head_size_qk = get<6>(params.problem_shape);
      auto &head_size_vo = get<7>(params.problem_shape);

      int num_chunks = cute::ceil_div(seq_len_kv, chunk_size);

    
      Tensor mQ_mkl = cute::get_xe_tensor(make_shape(seq_len_qo, head_size_qk,
                                                     (is_var_len ? 1 : batch) * num_heads_q)); //(m,k,l)

      for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
      {
        int current_chunk_size = cute::min(chunk_size,
                                           seq_len_kv - chunk_idx * chunk_size);

        Tensor mK_chunk_nkl = cute::get_xe_tensor(
            make_shape(current_chunk_size, head_size_qk,
                       (is_var_len ? 1 : batch) * num_head_kv)); //(n_chunk,k,l)

        Tensor mV_chunk_nkl = cute::get_xe_tensor(
            make_shape(head_size_vo, current_chunk_size,
                       (is_var_len ? 1 : batch) * num_head_kv)); //(n_chunk,k,l)

        Tensor mK_chunk_nk = mK_chunk_nkl(_, _, blk_l_coord / group_heads_q); // (n_chunk,k)
        Tensor mV_chunk_nk = mV_chunk_nkl(_, _, blk_l_coord / group_heads_q); // (n_chunk,k)

        auto gK_chunk = local_tile(mK_chunk_nk, TileShapeQK{},
                                   make_coord(_, _, _), Step<X, _1, _1>{});
        auto gV_chunk = local_tile(mV_chunk_nk, TileShapeOutput{},
                                   make_coord(_, blk_n_coord, _), Step<X, _1, _1>{});

        auto tiled_prefetch_k_chunk = cute::prefetch_selector<
            Shape<Int<QK_BLK_N>, Int<cute::max(cute::gcd(QK_BLK_K, 64), 32)>>,
            Num_SGs>(mainloop_params.gmem_tiled_copy_k);

        auto tiled_prefetch_v_chunk = cute::prefetch_selector<
            Shape<Int<cute::max(cute::gcd(Epilogue_BLK_N, 64), 32)>, Int<Epilogue_BLK_K>>,
            Num_SGs>(mainloop_params.gmem_tiled_copy_v);

        for (int nblock = 0; nblock < cute::ceil_div(current_chunk_size, QK_BLK_N); ++nblock)
        {
          barrier_arrive(barrier_scope);

          Tensor tSr = make_tensor<ElementAccumulator>(
              Shape<Int<Vec>, Int<FragsM>, Int<FragsN>>{});
          clear(tSr);

          collective_mma.mmaQK(tSr, gQ, gK_chunk(_, _, nblock, _), tSr,
                               ceil_div(head_size_qk, QK_BLK_K),
                               mainloop_params, false);

          CollectiveSoftmaxEpilogue softmax(params.softmax);
          softmax(chunk_idx == 0 && nblock == 0, tSr, max_reg, sum_reg, out_reg);

          collective_mma.template mmaPV<VSlicer>(
              out_reg, tSr, gV_chunk(_, _, nblock), out_reg,
              mainloop_params, false);

          // ... prefetch next tile ...

          barrier_wait(barrier_scope);
        }
      }

      // Epilogue
      auto epilogue_params = CollectiveEpilogue::template get_updated_copies<is_var_len>(
          params.epilogue, params.problem_shape, sequence_length_shape, batch_coord);
      CollectiveEpilogue epilogue{epilogue_params, shared_storage.epilogue};
      auto blk_coord_mnkl = make_coord(blk_m_coord, blk_n_coord, _, blk_l_coord);
      epilogue(params.problem_shape, sequence_length_shape, blk_coord_mnkl,
               out_reg, max_reg, sum_reg);
    }
  };

} // namespace cutlass::flash_attention::kernel