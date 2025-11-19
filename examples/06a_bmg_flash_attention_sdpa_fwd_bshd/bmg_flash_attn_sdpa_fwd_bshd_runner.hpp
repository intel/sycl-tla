/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/collective/xe_flash_attn_sdpa_fwd_bshd_epilogue.hpp"
#include "flash_attention_v2/collective/xe_flash_attn_sdpa_fwd_bshd_softmax_epilogue.hpp"
#include "flash_attention_v2/kernel/tile_scheduler_sdpa_fwd_bshd.hpp"
#include "flash_attention_v2/kernel/xe_sdpa_fwd_bshd.hpp"
#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "helper.h"
#include "sycl_common.hpp"

using namespace cute;

// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool is_causal;
  bool varlen = false;
  std::string scheduler;

  int batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, head_size_qk,
      head_size_vo, iterations;
  float softmax_scale;
  float scale;

  Options()
      : help(false), error(false), is_causal(false), varlen(false), batch(32),
        num_heads_q(16), num_heads_kv(16), seq_len_qo(512), head_size_qk(128),
        seq_len_kv(512), head_size_vo(128), iterations(100), scale(1.f),
        scheduler("Individual") {}

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    if (cmd.check_cmd_line_flag("is_causal")) {
      is_causal = true;
    }

    if (cmd.check_cmd_line_flag("varlen")) {
      varlen = true;
    }

    cmd.get_cmd_line_argument("scheduler", scheduler,
                              std::string("Individual"));

    cmd.get_cmd_line_argument("batch", batch, 32);
    cmd.get_cmd_line_argument("num_heads_q", num_heads_q, 16);
    cmd.get_cmd_line_argument("num_heads_kv", num_heads_kv, num_heads_q);
    cmd.get_cmd_line_argument("seq_len_qo", seq_len_qo, 512);
    cmd.get_cmd_line_argument("seq_len_kv", seq_len_kv, seq_len_qo);
    cmd.get_cmd_line_argument("head_size_vo", head_size_vo, HEAD_DIM);
    cmd.get_cmd_line_argument("head_size_qk", head_size_qk, head_size_vo);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("scale", scale);

    if (cmd.check_cmd_line_flag("scale")) {
      softmax_scale = scale;
    }
    else{
      softmax_scale = 1 / sqrt(static_cast<float>(head_size_qk));
    }
    
  }

  /// Prints the usage statement.
  std::ostream &print_usage(std::ostream &out) const {

    out << "BMG Flash Attention v2 Example\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage "
           "statement\n\n"
        << "  --is_causal                 Apply Causal Mask to the output of "
           "first Matmul\n"
        << "  --varlen                    Enable variable sequence length\n"
        << "  --scheduler=\"Value\"       Choose between Individual or "
           "Persistent Scheduler\n"
        << "  --batch=<int>               Sets the Batch Size of the "
           "Multi-Head Self Attention module\n"
        << "  --num_heads_q=<int>         Sets the Number of Attention Heads "
           "for Key-Value pair the Multi-Head Self Attention module\n"
        << "  --num_heads_kv=<int>        Sets the Number of Attention Heads "
           "for Query input in the Multi-Head Self Attention module\n"
        << "  --seq_len_qo=<int>          Sets the Sequence length of the "
           "Query input in Multi-Head Self Attention module\n"
        << "  --seq_len_kv=<int>          Sets the Sequence length of the "
           "Key-Value pair in Multi-Head Self Attention module\n"
        << "  --head_size_qk=<int>        Sets the Attention Head dimension of "
           "the 1st Matrix Multiplication in Multi-Head Self Attention module\n"
        << "  --head_size_vo=<int>        Sets the Attention Head dimension of "
           "the 2nd Matrix Multiplication in Multi-Head Self Attention module\n"
        << "  --iterations=<int>          Iterations\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Flash Attention takes 3 input matrices: (K)eys, (Q)ueries and (V)alues.
using LayoutQ = cutlass::layout::RowMajor;
using LayoutK = cutlass::layout::ColumnMajor;
using LayoutV = cutlass::layout::RowMajor;
using LayoutO = cutlass::layout::RowMajor;

template <class FMHAPrefillKernel, bool isVarLen> struct ExampleRunner {

  using StrideQ = typename FMHAPrefillKernel::StrideQ;
  using StrideK = typename FMHAPrefillKernel::StrideK;
  using StrideV = typename FMHAPrefillKernel::StrideV;
  using StrideO = typename FMHAPrefillKernel::StrideO;

  using ElementQ = typename FMHAPrefillKernel::ElementQ;
  using ElementK = typename FMHAPrefillKernel::ElementK;
  using ElementV = typename FMHAPrefillKernel::ElementV;
  using ElementAcc = typename FMHAPrefillKernel::ElementAccumulator;

  using CollectiveEpilogue = typename FMHAPrefillKernel::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using ProblemShapeType = typename FMHAPrefillKernel::ProblemShape;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementQ> block_Q;
  cutlass::DeviceAllocation<ElementK> block_K;
  cutlass::DeviceAllocation<ElementV> block_V;
  cutlass::DeviceAllocation<ElementOutput> block_O;
  cutlass::DeviceAllocation<ElementOutput> block_ref_O;
  cutlass::DeviceAllocation<float> block_LSE;
  cutlass::DeviceAllocation<float> block_ref_LSE;

  std::vector<int> cumulative_seqlen_q;
  std::vector<int> cumulative_seqlen_kv;
  cutlass::DeviceAllocation<int> device_cumulative_seqlen_q;
  cutlass::DeviceAllocation<int> device_cumulative_seqlen_kv;

  template <typename SrcT, typename DstT>
  void convert_fp8_to_fp16(const SrcT *d_src, DstT *d_dst, size_t size) {
    compat::get_default_queue()
        .parallel_for(
            size,
            [=](auto indx) { d_dst[indx] = static_cast<DstT>(d_src[indx]); })
        .wait();
  }

  template <typename T>
  static constexpr bool is_fp8_v =
      cute::is_any_of_v<T, cute::float_e5m2_t, cute::float_e4m3_t>;

  template <typename Tin>
  inline auto in_memory(cutlass::DeviceAllocation<Tin> &in) {
    using outType = cute::conditional_t<is_fp8_v<Tin>, half_t, Tin>;
    if constexpr (is_fp8_v<Tin>) {
      cutlass::DeviceAllocation<outType> out(in.size());
      convert_fp8_to_fp16<Tin, outType>(in.get(), out.get(), in.size());
      return out;
    } else {
      return in;
    };
  }
  //
  // Methods
  //
  bool verify(ProblemShapeType problem_size, bool is_causal, float softmax_scale) {
    if constexpr (isVarLen) {
      int max_seq_len_q = static_cast<int>(get<3>(problem_size));
      int max_seq_len_kv = static_cast<int>(get<4>(problem_size));
      get<3>(problem_size) = cutlass::fmha::collective::VariableLength{
          max_seq_len_q, cumulative_seqlen_q.data()};
      get<4>(problem_size) = cutlass::fmha::collective::VariableLength{
          max_seq_len_kv, cumulative_seqlen_kv.data()};
    }

    auto [batch, num_heads_q, num_heads_kv, head_size_qk, head_size_vo] =
        cute::select<0, 1, 2, 5, 6>(problem_size);
    int seq_len_qo, seq_len_kv;

    std::vector<ElementOutput> host_O(block_ref_O.size());
    std::vector<float> host_LSE(block_ref_LSE.size());
    auto block_Q_ = in_memory(block_Q);
    auto block_K_ = in_memory(block_K);
    auto block_V_ = in_memory(block_V);
    using ElementV_ = cute::conditional_t<is_fp8_v<ElementV>, half_t, ElementV>;

    int offset_q = 0;
    int offset_k = 0;
    int offset_v = 0;
    int offset_o = 0;
    int offset_lse = 0;

    int q_group_size = num_heads_q / num_heads_kv;
    // loop over the batch dimension to compute the output
    // to avoid the risk of running out of device memory
    for (int b = 0; b < batch; b++) {
      if constexpr (isVarLen) {
        auto logical_problem_shape =
            cutlass::fmha::collective::apply_variable_length(problem_size, b);
        seq_len_qo = get<3>(logical_problem_shape);
        seq_len_kv = get<4>(logical_problem_shape);
      } else {
        seq_len_qo = get<3>(problem_size);
        seq_len_kv = get<4>(problem_size);
      }

      // Initialize starting pointers for extrcating one Head * HeadDim from the
      // BSHD layout
      ElementQ *q_ptr;
      ElementK *k_ptr;
      ElementV *v_ptr;

      q_ptr = block_Q.get() + offset_q;
      k_ptr = block_K.get() + offset_k;
      v_ptr = block_V.get() + offset_v;

      for (int q_group = 0; q_group < num_heads_q / q_group_size; q_group++) {
        for (int q_head = 0; q_head < q_group_size; q_head++) {
          cutlass::DeviceAllocation<ElementAccumulator> block_S;
          block_S.reset(seq_len_qo * seq_len_kv);

          cutlass::TensorRef ref_Q(
              q_ptr,
              LayoutQ(num_heads_q *
                      head_size_qk)); // define the pitch - stride for next row
          cutlass::TensorRef ref_K(k_ptr, LayoutK(num_heads_kv * head_size_qk));
          cutlass::TensorRef ref_V(v_ptr, LayoutV(num_heads_kv * head_size_vo));
          cutlass::TensorRef ref_S(block_S.get(),
                                   LayoutQ::packed({seq_len_qo, seq_len_kv}));

          cutlass::reference::device::GemmComplex(
              {seq_len_qo, seq_len_kv, head_size_qk}, ElementAccumulator{1},
              ref_Q, cutlass::ComplexTransform::kNone, ref_K,
              cutlass::ComplexTransform::kNone, ElementAccumulator{0}, ref_S,
              ref_S, ElementAccumulator(0),
              1,                         // batch_count
              seq_len_qo * head_size_qk, // batch_stride_Q
              seq_len_kv * head_size_qk, // batch_stride_K
              seq_len_qo * seq_len_kv,   // batch_stride_S
              seq_len_qo * seq_len_kv    // batch_stride_S
          );
          compat::wait();
          std::vector<ElementAccumulator> host_S(block_S.size());
          compat::memcpy<ElementAccumulator>(host_S.data(), block_S.get(),
                                                 host_S.size());

          // delete this memory as it is no longer needed
          block_S.reset();

          // apply mask to S
          if (is_causal) {
            for (int row = 0; row < seq_len_qo; row++) {
              for (int col = 0; col < seq_len_kv; col++) {
                // Apply bottom right masking
                  // if (col > row - first_non_masked_sequence)
                  //   host_S[col + row * seq_len_kv] =
                  //       ElementAccumulator{-INFINITY};
                  if (seq_len_kv > seq_len_qo){
                    int first_masked_col_index = seq_len_kv - (seq_len_kv - seq_len_qo - 1) + row; // Find where does the masking occur for that sequence
                    if (col >= first_masked_col_index){
                      host_S[col + row * seq_len_kv] = ElementAccumulator{-INFINITY};
                    }
                  }
                  else {
                    if (seq_len_qo > seq_len_kv){
                      int first_non_masked_sequence = seq_len_qo - seq_len_kv;
                      if (row < first_non_masked_sequence || col > row - first_non_masked_sequence){
                        host_S[col + row * seq_len_kv] = ElementAccumulator{-INFINITY};
                      }
                    }
                    // seq_len_qo == seq_len_kv
                    else{

                    }
                  }

              }
            }
          }

          // compute max element per row of S
          std::vector<ElementAccumulator> max_vec(
              seq_len_qo, ElementAccumulator{-INFINITY});
          for (int row = 0; row < seq_len_qo; row++) {
            int idx = row * seq_len_kv;
            int max_idx = row;
            max_vec[max_idx] = host_S[idx++];
            for (int col = 1; col < seq_len_kv; col++, idx++) {
              if (max_vec[max_idx] < host_S[idx])
                max_vec[max_idx] = host_S[idx];
            }
          }

          // calculate scaled qk score for LSE
          std::vector<ElementAccumulator> qkscore_scaled(host_S.size(), ElementAccumulator{0});
          for (int row = 0; row < seq_len_qo; row++){
            for (int col = 0; col < seq_len_kv; col++) {
              qkscore_scaled[col + row * seq_len_kv] = host_S[col + row * seq_len_kv] * softmax_scale; // (qkscore * scale) - max_row
            }
          }

          // compute max element per row of scaled qk
          std::vector<ElementAccumulator> max_scaled_lse_vec(
              seq_len_qo, ElementAccumulator{-INFINITY});
          for (int row = 0; row < seq_len_qo; row++) {
            int idx = row * seq_len_kv;
            int max_idx = row;
            max_scaled_lse_vec[max_idx] = qkscore_scaled[idx++];
            for (int col = 1; col < seq_len_kv; col++, idx++) {
              if (max_scaled_lse_vec[max_idx] < qkscore_scaled[idx])
                max_scaled_lse_vec[max_idx] = qkscore_scaled[idx];
            }
          }

          // subtract scaled qk by max_row
          for (int row = 0; row < seq_len_qo; row++){
            for (int col = 0; col < seq_len_kv; col++) {
              qkscore_scaled[col + row * seq_len_kv] = qkscore_scaled[col + row * seq_len_kv] - max_scaled_lse_vec[row]; // (qkscore * scale) - max_row
            }
          }

          // calculate LSE
          for (int row = 0; row < seq_len_qo; row++){
            ElementAccumulator sum_exp = 0;
            for (int col = 0; col < seq_len_kv; col++) {
              sum_exp += expf(qkscore_scaled[col + row * seq_len_kv]);
            }
            host_LSE[row + offset_lse]  = std::isnan(max_scaled_lse_vec[row] + logf(sum_exp)) ? 0 : host_LSE[row + offset_lse] = max_scaled_lse_vec[row] + logf(sum_exp);
          }

          // compute exp of S
          for (int row = 0; row < seq_len_qo; row++) {
            int idx = row * seq_len_kv;
            int max_idx = row;
            for (int col = 0; col < seq_len_kv; col++, idx++) {
              host_S[idx] = expf((host_S[idx] - max_vec[max_idx]) * softmax_scale);
            }
          }

          // compute sum per row of S
          std::vector<ElementAccumulator> sum_vec(seq_len_qo,
                                                  ElementAccumulator{0});
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
              if (is_causal){
                if (seq_len_kv > seq_len_qo){
                  // Find where does the masking occur for that sequence
                  int first_masked_col_index = seq_len_kv - (seq_len_kv - seq_len_qo - 1) + row; 
                  if (col >= first_masked_col_index){
                    host_S[idx] = 0;
                  }
                  else{
                    host_S[idx] /= sum_vec[sum_idx];
                  }
                }
                else {
                  if (seq_len_qo > seq_len_kv){
                      int first_non_masked_sequence = seq_len_qo - seq_len_kv;
                      if (row < first_non_masked_sequence || col > row - first_non_masked_sequence){
                        host_S[idx] = 0;
                      }
                      else{
                        host_S[idx] /= sum_vec[sum_idx];
                      }
                  }
                  // seq_len_qo == seq_len_kv
                  else{

                  }
                }
              }
              // non-causal
              else{
                host_S[idx] /= sum_vec[sum_idx];
              }


              //apply bottom right masking
              // if (is_causal && col > row - first_non_masked_sequence) {
              //   host_S[idx] = 0;
              // } else {
              //   host_S[idx] /= sum_vec[sum_idx];
              // }
            }
          }

          std::vector<ElementV_> host_P(host_S.size());
          for (int p = 0; p < host_P.size(); p++)
            host_P[p] = static_cast<ElementV_>(host_S[p]);

          cutlass::DeviceAllocation<ElementV_> block_P;
          block_P.reset(host_P.size());

          compat::memcpy<ElementV_>(block_P.get(), host_P.data(),
                                        host_P.size());

          cutlass::TensorRef ref_P(block_P.get(),
                                   LayoutQ::packed({seq_len_qo, seq_len_kv}));

          cutlass::DeviceAllocation<ElementAccumulator> block_acc;
          block_acc.reset(seq_len_qo * head_size_vo);
          cutlass::TensorRef ref_acc(
              block_acc.get(), LayoutO::packed({seq_len_qo, head_size_vo}));

          cutlass::reference::device::GemmComplex(
              {seq_len_qo, head_size_vo, seq_len_kv}, ElementAccumulator{1},
              ref_P, cutlass::ComplexTransform::kNone, ref_V,
              cutlass::ComplexTransform::kNone, ElementAccumulator{0}, ref_acc,
              ref_acc, ElementAccumulator{0},
              1,                         // batch_count
              seq_len_qo * seq_len_kv,   // batch_stride_P
              seq_len_kv * head_size_vo, // batch_stride_V
              seq_len_qo * head_size_vo, // batch_stride_O
              seq_len_qo * head_size_vo  // batch_stride_O
          );

          compat::wait();
          // delete this memory as it is no longer needed
          block_P.reset();

          std::vector<ElementAccumulator> vec_acc(block_acc.size());
          compat::memcpy<ElementAccumulator>(
              vec_acc.data(), block_acc.get(), vec_acc.size());

          // delete this memory as it is no longer needed
          block_acc.reset();
          for (int seq = 0; seq < seq_len_qo; seq++) {
            for (int hvo = 0; hvo < head_size_vo; hvo++) {
              int idx = offset_o + seq * num_heads_q * head_size_vo +
                        (q_group * q_group_size + q_head) * head_size_vo + hvo;
              host_O[idx] =
                  static_cast<ElementOutput>(vec_acc[seq * head_size_vo + hvo]);
            }
          }
          q_ptr += head_size_qk;
          offset_lse += seq_len_qo;
        } // end of q_group loop
        // shift 1 head for each q_group loop
        k_ptr += head_size_qk;
        v_ptr += head_size_vo;
      } // end of q_head loop

      // shift the ptr to next batch -- [B, S, H, D]
      offset_q += seq_len_qo * num_heads_q * head_size_qk;
      offset_k += seq_len_kv * num_heads_kv * head_size_qk;
      offset_v += seq_len_kv * num_heads_kv * head_size_vo;
      offset_o += seq_len_qo * num_heads_q * head_size_vo;
    } // end of batch loop

    compat::wait();
    compat::memcpy<ElementOutput>(block_ref_O.get(), host_O.data(),
                                      host_O.size());
    compat::wait();
    compat::memcpy<float>(block_ref_LSE.get(), host_LSE.data(),
                              host_LSE.size());

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
        block_ref_O.get(), block_O.get(), block_O.size(), ElementOutput{0.01f},
        ElementOutput{0.01f});

    // Check if the LSE output from the CUTLASS kernel and reference kernel are
    // equal or not
    bool passed_lse = cutlass::reference::device::BlockCompareRelativelyEqual(
        block_ref_LSE.get(), block_LSE.get(), block_LSE.size(), 0.01f, 0.01f);

    passed ? print("Passed Output Accuracy \n") : print("Failed Output Accuracy \n");
    passed_lse ? print("Passed LSE Accuracy \n") : print("Failed LSE Accuracy \n");

    if (!passed){
      print("\n ================================= \n");
      std::vector<float> device_O(block_ref_O.size());
      compat::wait();
      compat::memcpy<float>(device_O.data(), block_O.get(),
                                block_O.size());
      print("\n host_O \n");
      for (int i = 0; i < host_O.size(); i++){
        if (i != 0 && i % 64 == 0){
          print('\n');
        }
        print(host_O[i]);
        print(' ');

      }
      print("\n Device O \n");
      for (int i = 0; i < device_O.size(); i++){
        if (i != 0 && i % 64 == 0){
          print('\n');
        }
        print(device_O[i]);
        print(' ');
      }
    }


    return passed && passed_lse;
  }

  template <class ProblemShape>
  auto initialize_varlen(const ProblemShape &problem_size) {
    int num_batches = get<0>(problem_size);

    // generate Q as --b times
    //    gaussian (--Q, --Q / 2) sampled positive
    //    track cumulative
    std::mt19937 rng(0x202305151552ull);
    std::normal_distribution<double> dist_q(get<3>(problem_size),
                                            get<3>(problem_size) / 2);
    std::normal_distribution<double> dist_kv(get<4>(problem_size),
                                             get<4>(problem_size) / 2);

    // Use Cacheline Size to calculate alignment
    constexpr int cacheline_bytes = 64;
    constexpr int AlignmentQ =
        cacheline_bytes /
        sizeof(ElementQ); // Alignment of Q matrix in units of elements
    constexpr int AlignmentKV =
        cacheline_bytes /
        sizeof(ElementK); // Alignment of Kand V matrix in units of elements

    auto generate_positive_int = [](auto &dist, auto &gen) {
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
      int seqlen_q =
          cutlass::round_up(generate_positive_int(dist_q, rng), AlignmentQ);
      int seqlen_kv =
          cutlass::round_up(generate_positive_int(dist_kv, rng), AlignmentKV);

      total_seqlen_q += seqlen_q;
      total_seqlen_kv += seqlen_kv;

      max_seqlen_q = std::max(max_seqlen_q, seqlen_q);
      max_seqlen_kv = std::max(max_seqlen_kv, seqlen_kv);

      cumulative_seqlen_q.push_back(cumulative_seqlen_q.back() + seqlen_q);
      cumulative_seqlen_kv.push_back(cumulative_seqlen_kv.back() + seqlen_kv);
    }

    ProblemShape problem_size_for_init = problem_size;
    get<0>(problem_size_for_init) = 1;
    get<3>(problem_size_for_init) = total_seqlen_q;
    get<4>(problem_size_for_init) = total_seqlen_kv;

    ProblemShapeType problem_size_for_launch;

    get<3>(problem_size_for_launch) =
        cutlass::fmha::collective::VariableLength{max_seqlen_q};
    get<4>(problem_size_for_launch) =
        cutlass::fmha::collective::VariableLength{max_seqlen_kv};
    get<5>(problem_size_for_launch) = get<5>(problem_size);
    get<6>(problem_size_for_launch) = get<6>(problem_size);
    get<0>(problem_size_for_launch) = get<0>(problem_size);
    get<1>(problem_size_for_launch) = get<1>(problem_size);
    get<2>(problem_size_for_launch) = get<2>(problem_size);

    return cute::make_tuple(problem_size_for_init, problem_size_for_launch);
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  ProblemShapeType initialize(const Options &options) {
    auto problem_shape_in = cute::make_tuple(
        options.batch, options.num_heads_q, options.num_heads_kv,
        options.seq_len_qo, options.seq_len_kv, options.head_size_qk,
        options.head_size_vo);

    ProblemShapeType problem_shape;
    decltype(problem_shape_in) problem_size;

    if constexpr (isVarLen) {
      auto [problem_shape_init, problem_shape_launch] =
          initialize_varlen(problem_shape_in);
      problem_shape = problem_shape_launch;
      problem_size = problem_shape_init;
    } else {
      problem_size = problem_shape_in;
      problem_shape = problem_shape_in;
    }

    auto [batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv,
          head_size_qk, head_size_vo] = problem_size;

    stride_Q = cutlass::make_cute_packed_stride(
        StrideQ{},
        cute::make_shape(seq_len_qo, num_heads_q * head_size_qk, batch));
    stride_K = cutlass::make_cute_packed_stride(
        StrideK{},
        cute::make_shape(seq_len_kv, num_heads_kv * head_size_qk, batch));
    stride_V = cutlass::make_cute_packed_stride(
        StrideV{},
        cute::make_shape(head_size_vo * num_heads_kv, seq_len_kv, batch));
    stride_O = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(seq_len_qo, num_heads_q * head_size_vo, batch));

    block_Q.reset(static_cast<std::size_t>(batch) * num_heads_q * seq_len_qo *
                  head_size_qk);
    block_K.reset(static_cast<std::size_t>(batch) * num_heads_kv * seq_len_kv *
                  head_size_qk);
    block_V.reset(static_cast<std::size_t>(batch) * num_heads_kv * seq_len_kv *
                  head_size_vo);
    block_O.reset(static_cast<std::size_t>(batch) * num_heads_q * seq_len_qo *
                  head_size_vo);
    block_ref_O.reset(static_cast<std::size_t>(batch) * num_heads_q *
                      seq_len_qo * head_size_vo);
    block_LSE.reset(static_cast<std::size_t>(batch) * num_heads_q * seq_len_qo);
    block_ref_LSE.reset(static_cast<std::size_t>(batch) * num_heads_q *
                        seq_len_qo);

    initialize_block(block_Q, seed + 2023);
    initialize_block(block_K, seed + 2022);
    initialize_block(block_V, seed + 2021);

    //debug
    // std::vector<ElementQ> block_Q_host(block_Q.size());
    // for (int i = 0; i < block_Q.size(); i++){
    //   block_Q_host[i] = static_cast<ElementQ>(i % 4);
    // }
    // std::vector<ElementK> block_K_host(block_K.size());
    // for (int i = 0; i < block_K.size(); i++){
    //   block_K_host[i] = static_cast<ElementK>(i % 3);
    // }
    // std::vector<ElementV> block_V_host(block_V.size());
    // for (int i = 0; i < block_V.size(); i++){
    //   block_V_host[i] = static_cast<ElementV>(i % 2);
    // }

    // compat::wait();
    // compat::memcpy<ElementQ>(block_Q.get(), block_Q_host.data(), block_Q.size());
    // compat::wait();
    // compat::memcpy<ElementK>(block_K.get(), block_K_host.data(), block_K.size());
    // compat::wait();
    // compat::memcpy<ElementV>(block_V.get(), block_V_host.data(), block_V.size());
    
    if (!cumulative_seqlen_q.empty()) {
      device_cumulative_seqlen_q.reset(cumulative_seqlen_q.size());
      device_cumulative_seqlen_q.copy_from_host(cumulative_seqlen_q.data(),
                                                cumulative_seqlen_q.size());
    }
    if (!cumulative_seqlen_kv.empty()) {
      device_cumulative_seqlen_kv.reset(cumulative_seqlen_kv.size());
      device_cumulative_seqlen_kv.copy_from_host(cumulative_seqlen_kv.data(),
                                                 cumulative_seqlen_kv.size());
    }

    if constexpr (isVarLen) {
      get<3>(problem_shape).cumulative_length =
          device_cumulative_seqlen_q.get();
      get<4>(problem_shape).cumulative_length =
          device_cumulative_seqlen_kv.get();
    }

    return problem_shape;
  }

  // Note that the GemmUniversalAdapter currently doesn't support flash
  // attention, which is why this secondary `run` function is required to launch
  // the kernel.
  static void run(typename FMHAPrefillKernel::Params params) {
    dim3 const block = FMHAPrefillKernel::get_block_shape();
    dim3 const grid = FMHAPrefillKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAPrefillKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

// Launch parameters depend on whether SYCL compiler supports work-group scratch
// memory extension
#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
    using namespace compat::experimental;
    auto event = launch<cutlass::device_kernel<FMHAPrefillKernel>>(
        launch_policy{sycl_grid, sycl_block,
                      local_mem_size{static_cast<std::size_t>(smem_size)},
                      kernel_properties{sycl_exp::sub_group_size<
                          FMHAPrefillKernel::DispatchPolicy::SubgroupSize>}},
        params);
#else
    compat::experimental::launch_properties launch_props{
        sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        sycl::ext::oneapi::experimental::sub_group_size<
            FMHAPrefillKernel::DispatchPolicy::SubgroupSize>};
    compat::experimental::launch_policy policy{sycl_grid, sycl_block,
                                                   launch_props, kernel_props};
    auto event = compat::experimental::launch<
        cutlass::device_kernel<FMHAPrefillKernel>>(policy, params);
#endif

    EventManager::getInstance().addEvent(event);
  }

  cutlass::Status run(const Options &options,
                      const cutlass::KernelHardwareInfo &hw_info) {

    ProblemShapeType problem_size = initialize(options);

    typename FMHAPrefillKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {block_Q.get(), stride_Q, block_K.get(), stride_K, block_V.get(),
         stride_V},
        {options.softmax_scale},
        {block_O.get(), stride_O, block_LSE.get()},
        hw_info,
        options.softmax_scale};
    
    // Define device-global scratch memory
    size_t workspace_size = FMHAPrefillKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (!FMHAPrefillKernel::can_implement(arguments)) {
      std::cout << "Invalid Problem Size: " << options.batch << 'x'
                << options.num_heads_q << 'x' << options.seq_len_qo << 'x'
                << options.seq_len_kv << 'x' << options.head_size_qk << 'x'
                << options.head_size_vo
                << (options.is_causal ? "xCausal" : "xNonCausal") << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Initialize the workspace
    CUTLASS_CHECK(
        FMHAPrefillKernel::initialize_workspace(arguments, workspace.get()));

    // Convert host-side arguments to device-side arguments to be passed to the
    // kernel
    auto params =
        FMHAPrefillKernel::to_underlying_arguments(arguments, workspace.get());

    // Run the GEMM
    run(params);

    compat::wait();

    // Verify that the result is correct
    bool passed = verify(problem_size, options.is_causal, options.softmax_scale);
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

    if (!passed) {
      return cutlass::Status::kErrorInternal;
    }

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        run(params);
      }
      compat::wait();
      // when seq_len_qo is not equal to seq_len_kv we use bottom up approach
      // for the masking. Following changes will adjust the effective_seq_len_kv
      // when masking applied for such cases
      auto offset = cute::min(options.seq_len_qo, options.seq_len_kv);
      auto discard_seq_coord = options.seq_len_qo - offset;
      auto full_tile_offset = options.seq_len_kv - offset;
      // offset + 1 is going to be ceil_div
      auto effective_seq_len_kv = options.is_causal
                                      ? full_tile_offset + ((offset + 1) / 2.0)
                                      : options.seq_len_kv;
      auto effective_seq_len_qo = options.is_causal
                                      ? options.seq_len_qo - discard_seq_coord
                                      : options.seq_len_qo;
      double cute_time = timer.seconds() / options.iterations;
      double flops_qk = 2.0 * options.batch * options.num_heads_q *
                        effective_seq_len_qo * effective_seq_len_kv *
                        options.head_size_qk;
      double flops_pv = 2.0 * options.batch * options.num_heads_q *
                        effective_seq_len_qo * options.head_size_vo *
                        effective_seq_len_kv;
      double tflops = ((flops_qk + flops_pv) * 1e-12) / cute_time;
      double gbps_qk =
          options.batch * (sizeof(ElementQ) * options.num_heads_q *
                               effective_seq_len_qo * options.head_size_qk +
                           sizeof(ElementK) * options.num_heads_kv *
                               effective_seq_len_kv * options.head_size_qk);
      double gbps_pv = sizeof(ElementV) * options.batch * options.num_heads_kv *
                           effective_seq_len_kv * options.head_size_vo +
                       sizeof(ElementOutput) * options.batch *
                           options.num_heads_q * effective_seq_len_qo *
                           options.head_size_vo;
      double gbps = ((gbps_qk + gbps_pv) * 1e-9) / (cute_time);
      std::cout << "Batch: " << options.batch
                << "\tNumHeads_q: " << options.num_heads_q
                << "\tNumHeads_kv: " << options.num_heads_kv
                << "\tSeq Length QO: " << options.seq_len_qo
                << "\tSeq Length KV: " << options.seq_len_kv
                << "\tHead Size QK: " << options.head_size_qk
                << "\tHead Size VO: " << options.head_size_vo
                << "\tCausal Mask: " << (options.is_causal ? "true" : "false")
                << "\tVariable Sequence Length: "
                << (options.varlen ? "true" : "false")
                << "\t Scheduler: " << options.scheduler;
      printf("\nPerformance:   %4.3f  GB/s,    %4.3f  TFlop/s,   %6.4f  ms\n\n",
             gbps, tflops, cute_time * 1000);
    }

    return cutlass::Status::kSuccess;
  }
};
// the default value used for the case BF16
template <
    bool Causal, typename TileShapeQK, typename TileShapePV,
    typename TileShapeOutput, typename SubgroupLayout, int PipelineStages,
    typename ElementInputQ = bfloat16_t, typename ElementInputKV = bfloat16_t,
    typename MMAOperation = XE_8x16x16_F32BF16BF16F32_TT,
    typename GmemTiledCopyQ = XE_2D_U16x8x32_LD_N,
    typename GmemTiledCopyK =
        XE_2D_U16x16x16_LD_T, // _T designates a transposed block load operation
    typename GmemTiledCopyV = XE_2D_U16x16x32_LD_V,
    typename ElementAccumulator = float,
    typename ElementComputeEpilogue = float, typename ElementOutput = float,
    typename GmemTiledCopyStore = XE_2D_U32x8x16_ST_N>
struct FMHAConfig {

  template <bool isVarLen, class Scheduler>
  static int run(const Options &options) {
    //
    // Run examples
    //

    // The KernelHardwareInfo struct holds the number of EUs on the GPU with a
    // given device ID. This information is used by the underlying kernel.
    cutlass::KernelHardwareInfo hw_info;

    // The code section below describes datatype for input, output matrices and
    // computation between elements in input matrices.

    using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;
    using GEMMDispatchPolicy =
        cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
    using CollectiveEpilogue =
        cutlass::flash_attention::collective::FlashPrefillEpilogue<
            EpilogueDispatchPolicy, MMAOperation, TileShapeOutput,
            SubgroupLayout, ElementComputeEpilogue, ElementOutput,
            cutlass::gemm::TagToStrideC_t<LayoutO>, ElementOutput,
            GmemTiledCopyStore>;
    using CollectiveSoftmaxEpilogue =
        cutlass::flash_attention::collective::FlashPrefillSoftmaxEpilogue<
            Causal, EpilogueDispatchPolicy, ElementAccumulator>;

    using ProblemShapeRegular = cute::tuple<int, int, int, int, int, int, int>;
    using namespace cutlass::fmha::collective;
    using ProblemShapeVarlen =
        cute::tuple<int, int, int, VariableLength, VariableLength, int, int>;
    using ProblemShapeType =
        std::conditional_t<isVarLen, ProblemShapeVarlen, ProblemShapeRegular>;

    // Mainloop
    using CollectiveMainloop =
        cutlass::flash_attention::collective::FlashPrefillMma<
            GEMMDispatchPolicy, ProblemShapeType, ElementInputQ,
            cutlass::gemm::TagToStrideA_t<LayoutQ>, ElementInputKV,
            cutlass::gemm::TagToStrideB_t<LayoutK>, ElementInputKV,
            cutlass::gemm::TagToStrideB_t<LayoutV>, MMAOperation, TileShapeQK,
            TileShapePV, SubgroupLayout,
            GmemTiledCopyQ, // Q
            GmemTiledCopyK, // K
            GmemTiledCopyV, // V,
            Causal>;

    using FMHAPrefillKernel = cutlass::flash_attention::kernel::FMHAPrefill<
        ProblemShapeType, CollectiveMainloop, CollectiveSoftmaxEpilogue,
        CollectiveEpilogue, Scheduler>;

    ExampleRunner<FMHAPrefillKernel, isVarLen> runner;

    CUTLASS_CHECK(runner.run(options, hw_info));
    return 0;
  }

  static int run(const Options &options) {
    if (options.varlen) {
      return run<true, cutlass::flash_attention::IndividualScheduler>(options);
    } else {
      return run<false, cutlass::flash_attention::IndividualScheduler>(options);
    }
  }
};