/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
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
    \brief Tests for device-wide Flash Attention Prefill interface
*/

#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "flash_attention_v2/kernel/xe_flash_attn_prefill.hpp"
#include "flash_attention_v2/collective/xe_flash_attn_prefill_epilogue.hpp"
#include "flash_attention_v2/collective/xe_flash_attn_prefill_softmax_epilogue.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/initialize_block.hpp"

#include <cute/tensor.hpp>
#include <random>
#include <cmath>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/sycl_tensor_fill.h"

#include "../gemm/device/testbed_utils.h"
#include "../common/cutlass_unit_test.h"

namespace test {
namespace flash_attention {

using namespace cute;

using MMAOperationBF16 = cute::XE_8x16x16_F32BF16BF16F32_TT;
using MMAOperationFP16 = cute::XE_8x16x16_F32F16F16F32_TT;

struct Shape_h64 {
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>;
};

struct Shape_h96 {
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>; 
};

struct Shape_h128 {
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128, _64>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>;
};

struct Shape_h192 {
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>; 
};

/////////////////////////////////////////////////////////////////////
  template <int input_bits, int output_bits> struct TiledCopyConfig;

  template <> struct TiledCopyConfig<8, 32> {
    using GmemTiledCopyQ = cute::XE_2D_U8x8x32_LD_N;
    using GmemTiledCopyK = cute::XE_2D_U8x16x16_LD_T;
    using GmemTiledCopyV = cute::XE_2D_U8x32x32_LD_V;
    using GmemTiledCopyO = cute::XE_2D_U32x8x16_ST_N;
  };

  template <> struct TiledCopyConfig<8, 8> {
    using GmemTiledCopyQ = cute::XE_2D_U8x8x32_LD_N;
    using GmemTiledCopyK = cute::XE_2D_U8x16x16_LD_T;
    using GmemTiledCopyV = cute::XE_2D_U8x32x32_LD_V;
    using GmemTiledCopyO = cute::XE_2D_U8x8x16_ST_N;
  };

  template <> struct TiledCopyConfig<16, 32> {
    using GmemTiledCopyQ = cute::XE_2D_U16x8x32_LD_N;
    using GmemTiledCopyK = cute::XE_2D_U16x16x16_LD_T;
    using GmemTiledCopyV = cute::XE_2D_U16x16x32_LD_V;
    using GmemTiledCopyO = cute::XE_2D_U32x8x16_ST_N;
  };

  template <> struct TiledCopyConfig<16, 16> {
    using GmemTiledCopyQ = cute::XE_2D_U16x8x32_LD_N;
    using GmemTiledCopyK = cute::XE_2D_U16x16x16_LD_T;
    using GmemTiledCopyV = cute::XE_2D_U16x16x32_LD_V;
    using GmemTiledCopyO = cute::XE_2D_U16x8x16_ST_N;
  };
/////////////////////////////////////////////////////////////////////

template<typename ElementInputType, typename ElementAccumulatorType, typename ElementOutputType,  
        typename TileShapeQK, typename TileShapePV, typename TileShapeOutput, typename SubgroupLayout, 
        typename MMAOperation, bool HasCausalMask, bool isVarLen, int PipelineStages, bool rope_enabled=false>
struct XE_Flash_Attention_Prefill {
  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;

  using ElementAccumulator = ElementAccumulatorType;
  using ElementComputeEpilogue = ElementAccumulatorType;
  using ElementInputQ = ElementInputType;
  using ElementInputKV = ElementInputType;
  using ElementOutput = ElementOutputType;

  using ProblemShapeRegular = cute::tuple<int, int, int, int, int, int, int>;
  using ProblemShapeVarlen = cute::tuple<int, int, int, cutlass::fmha::collective::VariableLength,
                                         cutlass::fmha::collective::VariableLength, int, int>;
  using ProblemShapeType = std::conditional_t<isVarLen, ProblemShapeVarlen, ProblemShapeRegular>;

  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  using GmemTiledCopyQ = typename TiledCopyConfig<cute::sizeof_bits_v<ElementInputQ>, cute::sizeof_bits_v<ElementOutput>>::GmemTiledCopyQ;
  using GmemTiledCopyK = typename TiledCopyConfig<cute::sizeof_bits_v<ElementInputKV>, cute::sizeof_bits_v<ElementOutput>>::GmemTiledCopyK;
  using GmemTiledCopyV = typename TiledCopyConfig<cute::sizeof_bits_v<ElementInputKV>, cute::sizeof_bits_v<ElementOutput>>::GmemTiledCopyV;
  using GmemTiledCopyStore = typename TiledCopyConfig<cute::sizeof_bits_v<ElementInputQ>, cute::sizeof_bits_v<ElementOutput>>::GmemTiledCopyO;
  using CollectiveEpilogue = cutlass::flash_attention::collective::FlashPrefillEpilogue<
        EpilogueDispatchPolicy, MMAOperation, TileShapeOutput, SubgroupLayout, ElementAccumulator, ElementOutput, cutlass::gemm::TagToStrideC_t<LayoutO>, ElementOutput,
        GmemTiledCopyStore>;
  using CollectiveSoftmaxEpilogue = cutlass::flash_attention::collective::FlashPrefillSoftmaxEpilogue<
        HasCausalMask, EpilogueDispatchPolicy, ElementAccumulator>;

  // Mainloop
  using CollectiveMainloop = cutlass::flash_attention::collective::FlashPrefillMma<
        GEMMDispatchPolicy, ProblemShapeType, ElementInputQ,
        cutlass::gemm::TagToStrideA_t<LayoutQ>, ElementInputKV,
        cutlass::gemm::TagToStrideB_t<LayoutK>, ElementInputKV,
        cutlass::gemm::TagToStrideB_t<LayoutV>,
        MMAOperation, TileShapeQK, TileShapePV, SubgroupLayout,
        GmemTiledCopyQ, // Q
        GmemTiledCopyK, // K
        GmemTiledCopyV, // V,
        HasCausalMask, rope_enabled>;

  using Kernel = cutlass::flash_attention::kernel::FMHAPrefill<ProblemShapeType, CollectiveMainloop,
                                                      CollectiveSoftmaxEpilogue, CollectiveEpilogue>;
};
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename FlashAttention>
struct TestbedImpl {
  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;

  using StrideQ = typename FlashAttention::StrideQ;
  using StrideK = typename FlashAttention::StrideK;
  using StrideV = typename FlashAttention::StrideV;
  using StrideO = typename FlashAttention::StrideO;

  using ElementQ = typename FlashAttention::ElementQ;
  using ElementK = typename FlashAttention::ElementK;
  using ElementV = typename FlashAttention::ElementV;
  using ElementAcc = typename FlashAttention::ElementAccumulator;

  using CollectiveMainloop = typename FlashAttention::CollectiveMainloop;
  using CollectiveEpilogue = typename FlashAttention::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using ProblemShapeType = typename FlashAttention::ProblemShape;
  static constexpr bool HasCausalMask = CollectiveMainloop::CausalMask;
  static constexpr bool isVarLen = CollectiveMainloop::is_var_len;

  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  uint64_t seed = 0;

  std::vector<int> cumulative_seqlen_q;
  std::vector<int> cumulative_seqlen_kv;
  cutlass::DeviceAllocation<int> device_cumulative_seqlen_q;
  cutlass::DeviceAllocation<int> device_cumulative_seqlen_kv;
  cutlass::DeviceAllocation<ElementQ> block_Q;
  cutlass::DeviceAllocation<ElementK> block_K;
  cutlass::DeviceAllocation<ElementV> block_V;
  cutlass::DeviceAllocation<ElementOutput> block_O;
  cutlass::DeviceAllocation<ElementOutput> block_ref_O;
  cutlass::DeviceAllocation<ElementQ> block_ref_Q;
  cutlass::DeviceAllocation<ElementK> block_ref_K;

  // RoPE support
  cutlass::DeviceAllocation<ElementQ> block_cos;
  cutlass::DeviceAllocation<ElementQ> block_sin;
  static constexpr bool rope_enabled = CollectiveMainloop::rope_enabled;

  //
  // Methods
  //

  template <typename SrcT, typename DstT>
  void convert_fp8_to_fp16(const SrcT* d_src, DstT* d_dst, size_t size) {
    syclcompat::get_default_queue().parallel_for(size, [=](auto indx) {
      d_dst[indx] = static_cast<DstT>(d_src[indx]);
    }).wait();
  }

  template <typename T>
  static constexpr bool is_fp8_v = cute::is_any_of_v<T, cute::float_e5m2_t, cute::float_e4m3_t>;

  template <typename Tin> inline auto in_memory(cutlass::DeviceAllocation<Tin>& in) {
    using outType = cute::conditional_t<is_fp8_v<Tin>, half_t, Tin>;
    if constexpr(is_fp8_v<Tin>) {
      cutlass::DeviceAllocation<outType> out(in.size());
      convert_fp8_to_fp16<Tin, outType>(in.get(), out.get(), in.size());
      return out;
    } else { 
      return in;
    };
  }

  /// Initialize RoPE cos/sin tensors
  void initialize_rope_tensors(int max_seq_len, int head_dim, int num_heads_q, int batch) {
    std::vector<ElementQ> cos_vals(max_seq_len * head_dim * num_heads_q * batch);
    std::vector<ElementQ> sin_vals(max_seq_len * head_dim * num_heads_q * batch);

    // fill data row-major wise
    for(int b = 0; b< num_heads_q*batch; b++){
      for (int pos = 0; pos < max_seq_len; ++pos) {
        for (int i = 0; i < head_dim/2 ; ++i) {
          int idx = b*max_seq_len*head_dim + pos*head_dim + 2*i;
          int idx1 = b*max_seq_len*head_dim + pos*head_dim + 2*i + 1;
          float theta = static_cast<float>(pos / std::pow(10000.0f, (2.0f * i) / head_dim));
          // float theta = i;
          cos_vals[idx] = static_cast<ElementQ>(std::cos(theta));
          cos_vals[idx1] = static_cast<ElementQ>(std::cos(theta));   
          sin_vals[idx] = static_cast<ElementQ>(std::sin(theta));
          sin_vals[idx1] = static_cast<ElementQ>(std::sin(theta));
        }
      }
    }
    syclcompat::memcpy(block_cos.get(), cos_vals.data(), cos_vals.size() * sizeof(ElementQ));
    syclcompat::memcpy(block_sin.get(), sin_vals.data(), sin_vals.size() * sizeof(ElementQ));
    syclcompat::wait();
  }

  /// Apply RoPE transformation to a tensor
  template<typename Element>
  void apply_rope_on_host(std::vector<Element>& tensor, int seq_len, int head_dim, int batch, int head,
                       const std::vector<ElementQ>& cos_vals, const std::vector<ElementQ>& sin_vals) {
      for (int seq_pos = 0; seq_pos < seq_len; ++seq_pos) {
        for (int dim_pair = 0; dim_pair < head_dim/2; ++dim_pair) {
          int cos_sin_idx = seq_pos * head_dim + dim_pair * 2;
          auto cos_val = static_cast<float>(cos_vals[cos_sin_idx]);
          auto sin_val = static_cast<float>(sin_vals[cos_sin_idx]);

          int x_idx = seq_pos * head_dim + dim_pair * 2;
          int y_idx = seq_pos * head_dim + dim_pair * 2 + 1;

          auto x = static_cast<float>(tensor[x_idx]);
          auto y = static_cast<float>(tensor[y_idx]);

          auto new_x = x * cos_val - y * sin_val;
          auto new_y = x * sin_val + y * cos_val;

          tensor[x_idx] = static_cast<Element>(new_x);
          tensor[y_idx] = static_cast<Element>(new_y);
        }
        
      }
  }

  template <class Element>
bool initialize_block_(
        cutlass::DeviceAllocation<Element>& block,
        uint64_t seed=2023, int seq_len=512, int head_dim=64, int batch_heads=64) {
  // creating a tensor of shape (batch_heads, seq_len, head_dim)
  // and filling it with sequential values for easy verification
  // e.g., for head_dim=4, seq_len=3, batch_heads=2
  // tensor[0, :, :] = [[0, 1, 2, 3],
  //                    [4, 5, 6, 7],
  //                    [8, 9, 10,11]]
  // tensor[1, :, :] = [[0, 1, 2, 3],
  //                    [4, 5, 6, 7],
  //                    [8, 9, 10,11]]
  // total elements = batch_heads * seq_len * head_dim
  // total elements = 2 * 3 * 4 = 24
  std::vector<Element> matrix(seq_len * head_dim*batch_heads);
  // cutlass::reference::device::BlockFillSequential(
  //      block.get(), block.size());

  // fill data row-major wise
  for(int i=0;i<batch_heads;i++){
    int temp = 0;
    for (int j = 0; j < seq_len; ++j) {
      for (int k = 0; k < head_dim ; k++) {
        
        int idx = i*seq_len*head_dim + j*head_dim + k;
        Element theta = static_cast<Element>(temp);
        matrix[idx] = theta;
        temp++;
      }
    }
  }
  syclcompat::wait();
  
  syclcompat::memcpy(block.get(), matrix.data(), matrix.size() * sizeof(Element));
  syclcompat::wait();

  return true;
}

  /// Initializes data structures
  template <class ProblemShape>
  ProblemShapeType initialize(ProblemShape problem_shape_in) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::initialize(problem_size)");
#endif
    ProblemShapeType problem_shape;
    ProblemShape problem_size;

    if constexpr (isVarLen) {
      auto [problem_shape_init, problem_shape_launch] = initialize_varlen(problem_shape_in);
      problem_shape = problem_shape_launch;
      problem_size = problem_shape_init;
    }
    else {
      problem_size = problem_shape_in;
      problem_shape = problem_shape_in;
    }

    auto [batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo] = problem_size;

    stride_Q = cutlass::make_cute_packed_stride(StrideQ{}, cute::make_shape(seq_len_qo, head_size_qk, batch * num_heads_q));
    stride_K = cutlass::make_cute_packed_stride(StrideK{}, cute::make_shape(seq_len_kv, head_size_qk, batch * num_heads_kv));
    stride_V = cutlass::make_cute_packed_stride(StrideV{}, cute::make_shape(head_size_vo, seq_len_kv, batch * num_heads_kv));
    stride_O = cutlass::make_cute_packed_stride(StrideO{}, cute::make_shape(seq_len_qo, head_size_vo, batch * num_heads_q));

    block_Q.reset(batch * num_heads_q * seq_len_qo * head_size_qk);
    block_K.reset(batch * num_heads_kv * seq_len_kv * head_size_qk);
    block_V.reset(batch * num_heads_kv * seq_len_kv * head_size_vo);
    block_O.reset(batch * num_heads_q * seq_len_qo * head_size_vo);
    block_ref_O.reset(batch * num_heads_q * seq_len_qo * head_size_vo);
    block_ref_Q.reset(batch * num_heads_q * seq_len_qo * head_size_qk);
    block_ref_K.reset(batch * num_heads_kv * seq_len_kv * head_size_qk);

    // Initialize RoPE tensors if enabled
    if constexpr (rope_enabled) {
      int max_seq_len = std::max(seq_len_qo, seq_len_kv);
      block_cos.reset(max_seq_len * head_size_qk * num_heads_q * batch);
      block_sin.reset(max_seq_len * head_size_qk * num_heads_q * batch);
      initialize_rope_tensors(max_seq_len, head_size_qk, num_heads_q, batch);
    }

    initialize_block(block_Q, seed + 2023);
    initialize_block(block_K, seed + 2022);
    initialize_block(block_V, seed + 2021);
    // initialize_block_(block_Q, seed + 2023, seq_len_qo, head_size_qk, batch * num_heads_q);
    // initialize_block_(block_K, seed + 2022, seq_len_kv, head_size_qk, batch * num_heads_kv);
    // initialize_block_(block_V, seed + 2021, head_size_vo, seq_len_kv, batch * num_heads_kv);
    syclcompat::wait();
    // reference copy of Q and K for verification
    syclcompat::memcpy(block_ref_Q.get(), block_Q.get(), block_Q.size() * sizeof(ElementQ));
    syclcompat::memcpy(block_ref_K.get(), block_K.get(), block_K.size() * sizeof(ElementK));
    syclcompat::wait();

    if (!cumulative_seqlen_q.empty()) {
      device_cumulative_seqlen_q.reset(cumulative_seqlen_q.size());
      device_cumulative_seqlen_q.copy_from_host(
        cumulative_seqlen_q.data(), cumulative_seqlen_q.size());
    }
    if (!cumulative_seqlen_kv.empty()) {
      device_cumulative_seqlen_kv.reset(cumulative_seqlen_kv.size());
      device_cumulative_seqlen_kv.copy_from_host(
        cumulative_seqlen_kv.data(), cumulative_seqlen_kv.size());
    }

    if constexpr (isVarLen) {
      cute::get<3>(problem_shape).cumulative_length = device_cumulative_seqlen_q.get();
      cute::get<4>(problem_shape).cumulative_length = device_cumulative_seqlen_kv.get();
    }

    return problem_shape;
  }

  template<class ProblemShape>
  auto initialize_varlen(const ProblemShape& problem_size, const bool VarlenSame = true) {
    int num_batches = cute::get<0>(problem_size);

    // generate Q as --b times
    //    gaussian (--Q, --Q / 2) sampled positive
    //    track cumulative
    std::mt19937 rng(0x202305151552ull);
    std::normal_distribution<double> dist_q(cute::get<3>(problem_size), cute::get<3>(problem_size) / 2);
    std::normal_distribution<double> dist_kv(cute::get<4>(problem_size), cute::get<4>(problem_size) / 2);

    // Use Cacheline Size to calculate alignment
    constexpr int cacheline_bytes = 64;
    constexpr int AlignmentQ = cacheline_bytes / sizeof(ElementQ);    // Alignment of Q matrix in units of elements
    constexpr int AlignmentKV = cacheline_bytes / sizeof(ElementK);   // Alignment of Kand V matrix in units of elements

    auto generate_positive_int = [](auto& dist, auto& gen) {
      int result = 0;
      do {
        result = static_cast<int>(dist(gen));
      } while (result <= 0);
      return result;
    };

    cumulative_seqlen_q = {0};
    cumulative_seqlen_kv = {0};

    int total_seqlen_q = 0;
    int total_seqlen_kv = 0;
    int max_seqlen_q = 0;
    int max_seqlen_kv = 0;

    for (int i = 0; i < num_batches; i++) {
      int seqlen_q = cutlass::round_up(generate_positive_int(dist_q, rng), AlignmentQ);
      int seqlen_kv = cutlass::round_up(generate_positive_int(dist_kv, rng), AlignmentKV);

      total_seqlen_q += seqlen_q;
      total_seqlen_kv += seqlen_kv;

      max_seqlen_q = std::max(max_seqlen_q, seqlen_q);
      max_seqlen_kv = std::max(max_seqlen_kv, seqlen_kv);

      cumulative_seqlen_q.push_back(cumulative_seqlen_q.back() + seqlen_q);
      cumulative_seqlen_kv.push_back(cumulative_seqlen_kv.back() + seqlen_kv);
    }

    ProblemShape problem_size_for_init = problem_size;
    cute::get<0>(problem_size_for_init) = 1;
    cute::get<3>(problem_size_for_init) = total_seqlen_q;
    cute::get<4>(problem_size_for_init) = total_seqlen_kv;

    ProblemShapeType problem_size_for_launch;

    cute::get<3>(problem_size_for_launch) = cutlass::fmha::collective::VariableLength{max_seqlen_q};
    cute::get<4>(problem_size_for_launch) = cutlass::fmha::collective::VariableLength{max_seqlen_kv};
    cute::get<5>(problem_size_for_launch) = cute::get<5>(problem_size);
    cute::get<6>(problem_size_for_launch) = cute::get<6>(problem_size);
    cute::get<0>(problem_size_for_launch) = cute::get<0>(problem_size);
    cute::get<1>(problem_size_for_launch) = cute::get<1>(problem_size);
    cute::get<2>(problem_size_for_launch) = cute::get<2>(problem_size);


    return cute::make_tuple(problem_size_for_init, problem_size_for_launch);
  }

  /// Verifies the result
  bool verify(ProblemShapeType problem_size, float softmax_scale)
  {
    if constexpr (isVarLen) {
      int max_seq_len_q = static_cast<int>(cute::get<3>(problem_size));
      int max_seq_len_kv = static_cast<int>(cute::get<4>(problem_size));
      cute::get<3>(problem_size) = cutlass::fmha::collective::VariableLength{max_seq_len_q, cumulative_seqlen_q.data()};
      cute::get<4>(problem_size) = cutlass::fmha::collective::VariableLength{max_seq_len_kv, cumulative_seqlen_kv.data()};
    }

    auto [batch, num_heads_q, num_heads_kv, head_size_qk, head_size_vo] = cute::select<0,1,2,5,6>(problem_size);
    int seq_len_qo, seq_len_kv;

    auto block_Q_ = in_memory(block_Q);
    auto block_K_ = in_memory(block_K);
    auto block_V_ = in_memory(block_V);
    auto block_cos_ = in_memory(block_cos);
    auto block_sin_ = in_memory(block_sin); 
    using ElementV_ = cute::conditional_t<is_fp8_v<ElementV>, half_t, ElementV>;

    int offset_q = 0;
    int offset_k = 0;
    int offset_v = 0;
    int offset_o = 0;
    // loop over the batch dimension to compute the output
    // to avoid the risk of running out of device memory
    int q_group_size = num_heads_q/num_heads_kv;
    for (int b = 0; b < batch; b++) {
      if constexpr (isVarLen) {
        auto logical_problem_shape = cutlass::fmha::collective::apply_variable_length(problem_size, b);
        seq_len_qo = cute::get<3>(logical_problem_shape);
        seq_len_kv = cute::get<4>(logical_problem_shape);
      } else {
        seq_len_qo = cute::get<3>(problem_size);
        seq_len_kv = cute::get<4>(problem_size);
      }
      int kv_group_update=1;
      for (int h = 0; h < num_heads_q; h++) {
        cutlass::DeviceAllocation<ElementAccumulator> block_S;
        block_S.reset(seq_len_qo * seq_len_kv);

        cutlass::TensorRef ref_Q(block_Q_.get() + offset_q, LayoutQ::packed({seq_len_qo, head_size_qk}));
        cutlass::TensorRef ref_K(block_K_.get() + offset_k, LayoutK::packed({head_size_qk, seq_len_kv}));
        cutlass::TensorRef ref_V(block_V_.get() + offset_v, LayoutV::packed({seq_len_kv, head_size_vo}));
        cutlass::TensorRef ref_S(block_S.get(), LayoutQ::packed({seq_len_qo, seq_len_kv}));

        // Apply RoPE to Q and K if enabled
        // if constexpr (rope_enabled) {
        //   cutlass::TensorRef ref_Q_cos(block_cos_.get() + offset_q, LayoutQ::packed({seq_len_qo, head_size_qk}));
        //   cutlass::TensorRef ref_Q_sin(block_sin_.get() + offset_q, LayoutQ::packed({seq_len_qo, head_size_qk}));
        //   cutlass::TensorRef ref_K_cos(block_cos_.get() + offset_k, LayoutK::packed({head_size_qk, seq_len_kv}));
        //   cutlass::TensorRef ref_K_sin(block_sin_.get() + offset_k, LayoutK::packed({head_size_qk, seq_len_kv}));

        //   std::vector<ElementQ> host_Q_cos(seq_len_qo* head_size_qk);
        //   std::vector<ElementQ> host_Q_sin(seq_len_qo* head_size_qk);

        //   std::vector<ElementQ> host_K_cos(head_size_qk* seq_len_kv);
        //   std::vector<ElementQ> host_K_sin(head_size_qk* seq_len_kv);

        //   syclcompat::memcpy(host_Q_cos.data(), ref_Q_cos.data() + offset_q, host_Q_cos.size() * sizeof(ElementQ));
        //   syclcompat::memcpy(host_Q_sin.data(), ref_Q_sin.data() + offset_q, host_Q_sin.size() * sizeof(ElementQ));
        //   syclcompat::memcpy(host_K_cos.data(), ref_K_cos.data() + offset_k, host_K_cos.size() * sizeof(ElementQ));
        //   syclcompat::memcpy(host_K_sin.data(), ref_K_sin.data() + offset_k, host_K_sin.size() * sizeof(ElementQ));
        //   syclcompat::wait();

        //   std::vector<ElementQ> host_Q(seq_len_qo* head_size_qk);
        //   std::vector<ElementK> host_K(head_size_qk* seq_len_kv);

        //   syclcompat::memcpy(host_Q.data(), ref_Q.data() , host_Q.size() * sizeof(ElementQ));
        //   syclcompat::memcpy(host_K.data(), ref_K.data() , host_K.size() * sizeof(ElementK));
        //   syclcompat::wait();

        //   // apply rope on host then update Q, K accordingly
        //   apply_rope_on_host(host_Q, seq_len_qo, head_size_qk, b, h, host_Q_cos, host_Q_sin);
        //   apply_rope_on_host(host_K, seq_len_kv, head_size_qk, b, h, host_Q_cos, host_Q_sin);
        //   syclcompat::wait();

        //   cutlass::DeviceAllocation<ElementQ> block_Q_rope;
        //   cutlass::DeviceAllocation<ElementK> block_K_rope;
        //   block_Q_rope.reset(host_Q.size());
        //   block_K_rope.reset(host_K.size());
          
        //   syclcompat::memcpy(block_Q_rope.get(), host_Q.data(), host_Q.size() * sizeof(ElementQ));
        //   syclcompat::memcpy(block_K_rope.get(), host_K.data(), host_K.size() * sizeof(ElementK));
        //   syclcompat::wait();
          
        //   // Update tensor references to use RoPE-transformed tensors 
        //   syclcompat::memcpy(ref_Q.data(), host_Q.data(), seq_len_qo* head_size_qk * sizeof(ElementQ));
        //   syclcompat::memcpy(ref_K.data(), host_K.data(), seq_len_kv* head_size_qk * sizeof(ElementK));
        // }

        cutlass::reference::device::GemmComplex({seq_len_qo, seq_len_kv, head_size_qk}, ElementAccumulator{1}, ref_Q,
                                                cutlass::ComplexTransform::kNone, ref_K, cutlass::ComplexTransform::kNone,
                                                ElementAccumulator{0}, ref_S, ref_S, ElementAccumulator{0},
                                                1,                   // batch_count
                                                seq_len_qo * head_size_qk, // batch_stride_Q
                                                seq_len_kv * head_size_qk, // batch_stride_K
                                                seq_len_qo * seq_len_kv,   // batch_stride_S
                                                seq_len_qo * seq_len_kv    // batch_stride_S
        );

        syclcompat::wait();

        std::vector<ElementAccumulator> host_S(block_S.size());
        syclcompat::memcpy<ElementAccumulator>(host_S.data(), block_S.get(), host_S.size());

        // delete this memory as it is no longer needed
        block_S.reset();
        auto offset = cute::min(seq_len_qo, seq_len_kv);
        auto discard_seq_coord = seq_len_qo - offset;
        auto full_tile_offset = seq_len_kv - offset;
        if (HasCausalMask) {
          // apply mask to S
          for (int row = 0; row < seq_len_qo; row++) {
            for (int col = 0; col < seq_len_kv; col++) {
              if ((col - full_tile_offset) > (row - discard_seq_coord))
                host_S[col + row * seq_len_kv] = ElementAccumulator{-INFINITY};
            }
          }
        }

        // compute max element per row of S
        std::vector<ElementAccumulator> max_vec(seq_len_qo, -INFINITY);
        for (int row = 0; row < seq_len_qo; row++) {
          int idx = row * seq_len_kv;
          int max_idx = row;
          max_vec[max_idx] = host_S[idx++];
          for (int col = 1; col < seq_len_kv; col++, idx++) {
            if (max_vec[max_idx] < host_S[idx])
              max_vec[max_idx] = host_S[idx];
          }
        }

        // compute exp of S
        for (int row = 0; row < seq_len_qo; row++) {
          int idx = row * seq_len_kv;
          int max_idx = row;
          for (int col = 0; col < seq_len_kv; col++, idx++) {
            host_S[idx] = expf((host_S[idx] - max_vec[max_idx]) / sqrt(static_cast<ElementAccumulator>((head_size_qk))));
          }
        }

        // compute sum per row of S
        std::vector<ElementAccumulator> sum_vec(seq_len_qo, ElementAccumulator{0});
        for (int row = 0; row < seq_len_qo; row++) {
          int idx = row * seq_len_kv;
          int sum_idx = row;
          for (int col = 0; col < seq_len_kv; col++, idx++) {
            sum_vec[sum_idx] += host_S[idx];
          }

          // scale each row with the sum to compute softmax
          idx = row * seq_len_kv;
          sum_idx = row;
          for (int col = 0; col < seq_len_kv; col++, idx++) {
            if(HasCausalMask && row < discard_seq_coord) {
              host_S[idx] = 0;
            } else {
              host_S[idx] /= sum_vec[sum_idx];
            }
          }
        }

        std::vector<ElementV_> host_P(host_S.size());
        for (int p = 0; p < host_P.size(); p++)
          host_P[p] = static_cast<ElementV_>(host_S[p]);

        cutlass::DeviceAllocation<ElementV_> block_P;
        block_P.reset(host_P.size());

        syclcompat::memcpy<ElementV_>(block_P.get(), host_P.data(), host_P.size());

        cutlass::TensorRef ref_P(block_P.get(), LayoutQ::packed({seq_len_qo, seq_len_kv}));

        cutlass::DeviceAllocation<ElementAccumulator> block_acc;
        block_acc.reset(seq_len_qo * head_size_vo);
        cutlass::TensorRef ref_acc(block_acc.get(), LayoutO::packed({seq_len_qo, head_size_vo}));

        cutlass::reference::device::GemmComplex({seq_len_qo, head_size_vo, seq_len_kv}, ElementAccumulator{1}, ref_P,
                                                cutlass::ComplexTransform::kNone, ref_V, cutlass::ComplexTransform::kNone,
                                                ElementAccumulator{0}, ref_acc, ref_acc, ElementAccumulator{0},
                                                1,                   // batch_count
                                                seq_len_qo * seq_len_kv,   // batch_stride_P
                                                seq_len_kv * head_size_vo, // batch_stride_V
                                                seq_len_qo * head_size_vo, // batch_stride_O
                                                seq_len_qo * head_size_vo  // batch_stride_O
        );

        syclcompat::wait();
        // delete this memory as it is no longer needed
        block_P.reset();

        std::vector<ElementAccumulator> vec_acc(block_acc.size());
        syclcompat::memcpy<ElementAccumulator>(vec_acc.data(), block_acc.get(), vec_acc.size());

        // delete this memory as it is no longer needed
        block_acc.reset();
        std::vector<ElementOutput> vec_out(vec_acc.size());
        for(int i = 0; i < vec_out.size(); i++) {
          vec_out[i] = static_cast<ElementOutput>(vec_acc[i]);
        }
        syclcompat::memcpy<ElementOutput>(block_ref_O.get() + offset_o, vec_out.data(), vec_out.size());

        offset_q += seq_len_qo * head_size_qk;
        if(kv_group_update % q_group_size==0) {
          offset_k += seq_len_kv * head_size_qk;
          offset_v += seq_len_kv * head_size_vo;
        }
        kv_group_update++;
        offset_o += seq_len_qo * head_size_vo;
      }
    }

    syclcompat::wait();
    std::vector<ElementOutput> host_ref_o(block_O.size());
    std::vector<ElementOutput> host_o(block_O.size());
    syclcompat::memcpy(host_ref_o.data(), block_ref_O.get(), block_ref_O.size() * sizeof(ElementOutput));
    syclcompat::memcpy(host_o.data(), block_O.get(), block_O.size() * sizeof(ElementOutput));
    syclcompat::wait();
    for(int i = 0; i < host_o.size(); i++) {
      std::cout << "O[" << i << "] = " << host_o[i] << ", ref_O[" << i << "] = " << host_ref_o[i] << ", diff : " << (host_o[i] - host_ref_o[i]) << std::endl;
    }

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(block_ref_O.get(), block_O.get(),
                                                                          block_O.size(), ElementOutput{0.5}, ElementOutput{0.5});
    return passed;
  }

  bool sufficient() {
    return true;
  }

  /// Executes one test
  template<class ProblemShape>
  bool run(ProblemShape problem_size_init, float softmax_scale)
  {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run"); 
#endif

    // Fail test if insufficient device
    if (!sufficient()) {
      CUTLASS_TRACE_HOST("TestbedImpl::run: Test failed due to insufficient device");
      std::cout << "Test failed due to insufficient device." << std::endl;
      return false;
    }
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    else {
      CUTLASS_TRACE_HOST("TestbedImpl::run: sufficient() returned true");
    }
#endif

    ProblemShapeType problem_size = this->initialize(problem_size_init);

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: this->initialize() returned true");
#endif

    //
    // Initialize the Flash attention operator
    //
    cutlass::KernelHardwareInfo hw_info;

    // Prepare mainloop arguments with RoPE tensors if enabled
    auto mainloop_args = [&]() {
      if constexpr (rope_enabled) {
        return typename FlashAttention::CollectiveMainloop::Arguments{
          block_Q.get(), stride_Q, 
          block_K.get(), stride_K, 
          block_V.get(), stride_V,
          block_cos.get(),
          block_sin.get(), 
          // stride_Q_cs, 
          // stride_K_cs,
        };
      } else {
        return typename FlashAttention::CollectiveMainloop::Arguments{
          block_Q.get(), stride_Q, 
          block_K.get(), stride_K, 
          block_V.get(), stride_V
        };
      }
    }();


    typename FlashAttention::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      mainloop_args,
      {softmax_scale},
      {block_O.get(), stride_O},
      hw_info};

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Calling FlashAttention::get_workspace_size");
#endif
    size_t workspace_size = FlashAttention::get_workspace_size(arguments);
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Allocating workspace of size " << workspace_size);
#endif
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Calling FlashAttention::can_implement");
#endif
    auto can_implement = FlashAttention::can_implement(arguments);

    if (!can_implement) {
      std::cerr << "This test is not supported." << "\n";
    }

    //
    // Run Flash attention
    //

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Calling to_underlying_arguments");
#endif
    auto params = FlashAttention::to_underlying_arguments(arguments, workspace.get());

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Calling run");
#endif
    auto const block = FlashAttention::get_block_shape();
    auto const grid = FlashAttention::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FlashAttention::SharedStorageSize;

    const auto sycl_block = syclcompat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = syclcompat::dim3(grid.x, grid.y, grid.z);

#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
    using namespace syclcompat::experimental;
    auto event = launch<cutlass::device_kernel<FlashAttention>>(
        launch_policy{sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)},
                      kernel_properties{sycl_exp::sub_group_size<FlashAttention::DispatchPolicy::SubgroupSize>}},
        params);
#else
    syclcompat::experimental::launch_properties launch_props {
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    syclcompat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<FlashAttention::DispatchPolicy::SubgroupSize>
    };
    syclcompat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
    auto event = syclcompat::experimental::launch<cutlass::device_kernel<FlashAttention>>(policy, params);
#endif
    EventManager::getInstance().addEvent(event);

    try {
      syclcompat::wait_and_throw();
    } catch (std::exception const &e) {
      ADD_FAILURE() << "Error at Kernel Sync.";
      return false;
    }

    //
    // Verify
    //
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Calling this->verify");
#endif
    bool passed = this->verify(problem_size, softmax_scale);
    if (!passed) {
      CUTLASS_TRACE_HOST("TestbedImpl::run: this->verify FAILED");
      std::cout << "Error : Failed \n";
    }
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    else {
      CUTLASS_TRACE_HOST("TestbedImpl::run: this->verify passed");
    }
#endif

#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("TestbedImpl::run: Reached end");
#endif
    return passed;
  }
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename FlashAttention
>
struct Testbed3x {
  using TestBedImpl = typename detail::TestbedImpl<FlashAttention>;
  TestBedImpl impl_;

  //
  // Methods
  //
  Testbed3x() : impl_() {}

  /// Executes one test
  template <class ProblemShape>
  bool run(
   ProblemShape problem_size,
   float softmax_scale
    )
  {
    return impl_.run(problem_size, softmax_scale);
  }
};

template <typename FlashAttention>
bool TestFlashPrefillAll(int head_size, std::string config="default") {
  Testbed3x<FlashAttention> testbed;

  std::vector<int> problem_size_batch;
  std::vector<int> problem_size_num_heads;
  std::vector<int> problem_size_seq_len;

  if(config == "llama3_70b"){
    problem_size_batch = {1, 2};
    problem_size_num_heads = {128};
    problem_size_seq_len = {512, 1024};
  }
  else{
    problem_size_batch = {8};
    problem_size_num_heads = {8};
    problem_size_seq_len = {512};
  }
  std::vector<float> problem_size_softmax_scale{ 1.f / sqrt(static_cast<float>(head_size)) };
  bool passed = true;

  for (int batch : problem_size_batch) {
    for (int num_heads : problem_size_num_heads) {
      for (int seq_len : problem_size_seq_len) {
        for (float softmax_scale : problem_size_softmax_scale) {
          auto num_heads_q = num_heads;
          auto num_heads_kv = num_heads;
          auto seq_len_qo = seq_len;
          auto seq_len_kv = seq_len;
          auto head_size_qk = head_size;
          auto head_size_vo = head_size;

          auto problem_size = cute::make_tuple(
            batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo);
          try {
            passed = testbed.run(problem_size, softmax_scale);
          }
          catch (std::exception const& e) {
            EXPECT_TRUE(false) << "TestAll: testbed.run {"
              << "batch: " << batch << ", num_heads_q: " << num_heads_q << ", num_heads_kv: " << num_heads_kv
              << ", seq_len_qo: " << seq_len_qo << ", seq_len_kv: " << seq_len_kv
              << ", head_size_vo: " << head_size_vo << ", head_size_qk: " << head_size_qk
              << ", scale: " << softmax_scale
              << "} threw an exception: " << e.what();
            throw;
          }
          catch (...) {
            EXPECT_TRUE(false) << "TestAll: testbed.run {"
              << "batch: " << batch << ", num_heads_q: " << num_heads_q << ", num_heads_kv: " << num_heads_kv
              << ", seq_len_qo: " << seq_len_qo << ", seq_len_kv: " << seq_len_kv
              << ", head_size_vo: " << head_size_vo << ", head_size_qk: " << head_size_qk
              << ", scale: " << softmax_scale
              << "} threw an exception (unknown)";
            throw;
          }

          EXPECT_TRUE(passed) << "TestAll: testbed.run {"
            << "batch: " << batch << ", num_heads_q: " << num_heads_q << ", num_heads_kv: " << num_heads_kv
            << ", seq_len_qo: " << seq_len_qo << ", seq_len_kv: " << seq_len_kv
            << ", head_size_vo: " << head_size_vo << ", head_size_qk: " << head_size_qk
            << ", scale: " << softmax_scale
            << "} failed";

          if (!passed) {
            std::cout << __FILE__ << ':' << __LINE__ << " : Flash attention FAILED.\n";
            return false;
          }
        } // softmax_scale
      } // seq_len
    } // num_heads
  }  // batch
  return passed;
}

} // namespace flash_attention
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
