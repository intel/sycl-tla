/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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
#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/kernel/xe_fhma_fwd_kernel.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include <cute/tensor.hpp>
#include <random>

#include "helper.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

using namespace cute;

// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool is_causal;
  bool varlen = false;
  std::string scheduler;

  int batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo, iterations;
  float softmax_scale;

  Options()
      : help(false), error(false), is_causal(false), varlen(false), batch(32), num_heads_q(16), num_heads_kv(16), seq_len_qo(512), head_size_qk(128),
        seq_len_kv(512), head_size_vo(128), iterations(100), softmax_scale(1.f), scheduler("Individual") {}

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

    cmd.get_cmd_line_argument("scheduler", scheduler, std::string("Individual"));

    cmd.get_cmd_line_argument("batch", batch, 32);
    cmd.get_cmd_line_argument("num_heads_q", num_heads_q, 16);
    cmd.get_cmd_line_argument("num_heads_kv", num_heads_kv, num_heads_q);
    cmd.get_cmd_line_argument("seq_len_kv", seq_len_kv, 512);
#ifdef DECODE
    cmd.get_cmd_line_argument("seq_len_qo", seq_len_qo, 1);
#else
    cmd.get_cmd_line_argument("seq_len_qo", seq_len_qo, seq_len_kv);
#endif
    cmd.get_cmd_line_argument("head_size_vo", head_size_vo, HEAD_DIM);
    cmd.get_cmd_line_argument("head_size_qk", head_size_qk, head_size_vo);
    cmd.get_cmd_line_argument("iterations", iterations, 100);

    softmax_scale = 1 / sqrt(static_cast<float>(head_size_qk));
  }

  /// Prints the usage statement.
  std::ostream &print_usage(std::ostream &out) const {

    out << "Xe FMHA Example\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage statement\n\n"
        << "  --is_causal                 Apply Causal Mask to the output of first Matmul\n"
        << "  --varlen                    Enable variable sequence length\n"
        << "  --scheduler=\"Value\"       Choose between Individual or Persistent Scheduler\n"
        << "  --batch=<int>               Sets the Batch Size of the Multi-Head Self Attention module\n"
        << "  --num_heads_q=<int>         Sets the Number of Attention Heads for Key-Value pair the Multi-Head Self Attention module\n"
        << "  --num_heads_kv=<int>        Sets the Number of Attention Heads for Query input in the Multi-Head Self Attention module\n"
        << "  --seq_len_qo=<int>          Sets the Sequence length of the Query input in Multi-Head Self Attention module\n"
        << "  --seq_len_kv=<int>          Sets the Sequence length of the Key-Value pair in Multi-Head Self Attention module\n"
        << "  --head_size_qk=<int>        Sets the Attention Head dimension of the 1st Matrix Multiplication in Multi-Head Self Attention module\n"
        << "  --head_size_vo=<int>        Sets the Attention Head dimension of the 2nd Matrix Multiplication in Multi-Head Self Attention module\n"
        << "  --iterations=<int>          Iterations\n\n";

    return out;
  }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers

template <typename SrcT, typename DstT> class ConvertTensorKernelTag{};

template <typename SrcT, typename DstT>
void convert_tensor(const SrcT* d_src, DstT* d_dst, size_t size) {
  using Tag = ConvertTensorKernelTag<SrcT, DstT>;
  compat::get_default_queue().parallel_for<Tag>(size, [=](auto indx) {
    d_dst[indx] = static_cast<DstT>(d_src[indx]);
  }).wait();
}

template <typename InT> inline auto in_memory(cutlass::DeviceAllocation<InT>& in) {
  using OutT = cute::conditional_t<(sizeof_bits_v<InT> <= 8), half_t, InT>;
  if constexpr (!is_same_v<InT, OutT>) {
    cutlass::DeviceAllocation<OutT> out(in.size());
    convert_tensor<InT, OutT>(in.get(), out.get(), in.size());
    return out;
  } else {
    return in;
  };
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// 3 input matrices: (K)eys, (Q)ueries and (V)alues.
using LayoutQ = cutlass::layout::RowMajor;
using LayoutK = cutlass::layout::ColumnMajor;
using LayoutV = cutlass::layout::RowMajor;
using LayoutO = cutlass::layout::RowMajor;

template <class FMHAKernel> struct ExampleRunner {

  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;

  using ElementQ = typename FMHAKernel::ElementQ;
  using ElementK = typename FMHAKernel::ElementK;
  using ElementV = typename FMHAKernel::ElementV;
  using ElementO = typename FMHAKernel::ElementO;

  using CollectiveMainloop = typename FMHAKernel::CollectiveMainloop;
  using ElementS = typename CollectiveMainloop::ElementS;

  using ProblemShapeType = typename FMHAKernel::ProblemShape;

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
  cutlass::DeviceAllocation<ElementO> block_O;
  cutlass::DeviceAllocation<ElementO> block_ref_O;

  //
  // Methods
  //

  bool verify(ProblemShapeType shape, bool is_causal) {

    auto batch = shape.batch;
    auto num_heads_q = shape.num_heads_q;
    auto num_heads_kv = shape.num_heads_kv;
    auto head_size_qk = shape.head_size_qk;
    auto head_size_vo = shape.head_size_vo;
    auto seq_len_qo = shape.seq_len_qo;
    auto seq_len_kv = shape.seq_len_kv;

    auto block_Q_ = in_memory(block_Q);
    auto block_K_ = in_memory(block_K);
    auto block_V_ = in_memory(block_V);

    using ElementV_ = std::remove_pointer_t<decltype(block_V_.get())>;

    int offset_q = 0;
    int offset_k = 0;
    int offset_v = 0;
    int offset_o = 0;

    // loop over the batch dimension to compute the output
    // to avoid the risk of running out of device memory
    int q_group_size = num_heads_q/num_heads_kv;
    for (int b = 0; b < batch; b++) {
      int kv_group_update=1;
      for (int h = 0; h < num_heads_q; h++) {
        cutlass::DeviceAllocation<ElementS> block_S;
        block_S.reset(seq_len_qo * seq_len_kv);

        cutlass::TensorRef ref_Q(block_Q_.get() + offset_q, LayoutQ::packed({seq_len_qo, head_size_qk}));
        cutlass::TensorRef ref_K(block_K_.get() + offset_k, LayoutK::packed({head_size_qk, seq_len_kv}));
        cutlass::TensorRef ref_V(block_V_.get() + offset_v, LayoutV::packed({seq_len_kv, head_size_vo}));
        cutlass::TensorRef ref_S(block_S.get(), LayoutQ::packed({seq_len_qo, seq_len_kv}));

        cutlass::reference::device::GemmComplex({seq_len_qo, seq_len_kv, head_size_qk}, 1.f, ref_Q,
                                                cutlass::ComplexTransform::kNone, ref_K, cutlass::ComplexTransform::kNone,
                                                0.f, ref_S, ref_S, ElementS(0),
                                                1,                   // batch_count
                                                seq_len_qo * head_size_qk, // batch_stride_Q
                                                seq_len_kv * head_size_qk, // batch_stride_K
                                                seq_len_qo * seq_len_kv,   // batch_stride_S
                                                seq_len_qo * seq_len_kv    // batch_stride_S
        );

        compat::wait();

        std::vector<ElementS> host_S(block_S.size());
        compat::memcpy<ElementS>(host_S.data(), block_S.get(), host_S.size());

        // delete this memory as it is no longer needed
        block_S.reset();
        auto offset = cute::min(seq_len_qo, seq_len_kv);
        auto discard_seq_coord = seq_len_qo - offset;
        auto full_tile_offset = seq_len_kv - offset;
        if (is_causal) {
          // apply mask to S
          for (int row = 0; row < seq_len_qo; row++) {
            for (int col = 0; col < seq_len_kv; col++) {
              if ((col - full_tile_offset) > (row - discard_seq_coord))
                host_S[col + row * seq_len_kv] = ElementS{-INFINITY};
            }
          }
        }

        // compute max element per row of S
        std::vector<ElementS> max_vec(seq_len_qo, ElementS{-INFINITY});
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
            /* FIXME: use softmax_scale instead of assuming its value here */
            host_S[idx] = expf((host_S[idx] - max_vec[max_idx]) / sqrt(static_cast<ElementS>((head_size_qk))));
          }
        }

        // compute sum per row of S
        std::vector<ElementS> sum_vec(seq_len_qo, ElementS{0});
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
            if(is_causal && row < discard_seq_coord) {
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

        compat::memcpy<ElementV_>(block_P.get(), host_P.data(), host_P.size());

        cutlass::TensorRef ref_P(block_P.get(), LayoutQ::packed({seq_len_qo, seq_len_kv}));

        cutlass::DeviceAllocation<ElementS> block_acc;
        block_acc.reset(seq_len_qo * head_size_vo);
        cutlass::TensorRef ref_acc(block_acc.get(), LayoutO::packed({seq_len_qo, head_size_vo}));

        cutlass::reference::device::GemmComplex({seq_len_qo, head_size_vo, seq_len_kv}, ElementS{1}, ref_P,
                                                cutlass::ComplexTransform::kNone, ref_V, cutlass::ComplexTransform::kNone,
                                                ElementS{0}, ref_acc, ref_acc, ElementS{0},
                                                1,                   // batch_count
                                                seq_len_qo * seq_len_kv,   // batch_stride_P
                                                seq_len_kv * head_size_vo, // batch_stride_V
                                                seq_len_qo * head_size_vo, // batch_stride_O
                                                seq_len_qo * head_size_vo  // batch_stride_O
        );

        compat::wait();
        // delete this memory as it is no longer needed
        block_P.reset();

        std::vector<ElementS> vec_acc(block_acc.size());
        compat::memcpy<ElementS>(vec_acc.data(), block_acc.get(), vec_acc.size());

        // delete this memory as it is no longer needed
        block_acc.reset();
        std::vector<ElementO> vec_out(vec_acc.size());
        for(int i = 0; i < vec_out.size(); i++) {
          vec_out[i] = static_cast<ElementO>(vec_acc[i]);
        }
        compat::memcpy<ElementO>(block_ref_O.get() + offset_o, vec_out.data(), vec_out.size());

        offset_q += seq_len_qo * head_size_qk;
        if(kv_group_update % q_group_size==0) {
          offset_k += seq_len_kv * head_size_qk;
          offset_v += seq_len_kv * head_size_vo;
        }
        kv_group_update++;
        offset_o += seq_len_qo * head_size_vo;
      }
    }

    compat::wait();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(block_ref_O.get(), block_O.get(),
                                                                          block_O.size(), ElementO{0.005}, ElementO{0.005});

    return passed;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  ProblemShapeType initialize(const Options &options) {
    ProblemShapeType shape;
    auto batch        = shape.batch        = options.batch;
    auto num_heads_q  = shape.num_heads_q  = options.num_heads_q;
    auto num_heads_kv = shape.num_heads_kv = options.num_heads_kv;
    auto seq_len_qo   = shape.seq_len_qo   = options.seq_len_qo;
    auto seq_len_kv   = shape.seq_len_kv   = options.seq_len_kv;
    auto head_size_qk = shape.head_size_qk = options.head_size_qk;
    auto head_size_vo = shape.head_size_vo = options.head_size_vo;

    stride_Q = cutlass::make_cute_packed_stride(StrideQ{}, cute::make_shape(seq_len_qo, head_size_qk, num_heads_q,  batch));
    stride_K = cutlass::make_cute_packed_stride(StrideK{}, cute::make_shape(seq_len_kv, head_size_qk, num_heads_kv, batch));
    stride_V = cutlass::make_cute_packed_stride(StrideV{}, cute::make_shape(head_size_vo, seq_len_kv, num_heads_kv, batch));
    stride_O = cutlass::make_cute_packed_stride(StrideO{}, cute::make_shape(seq_len_qo, head_size_vo, num_heads_q,  batch));

    block_Q.reset(static_cast<std::size_t>(batch) * num_heads_q * seq_len_qo * head_size_qk);
    block_K.reset(static_cast<std::size_t>(batch) * num_heads_kv * seq_len_kv * head_size_qk);
    block_V.reset(static_cast<std::size_t>(batch) * num_heads_kv * seq_len_kv * head_size_vo);
    block_O.reset(static_cast<std::size_t>(batch) * num_heads_q * seq_len_qo * head_size_vo);
    block_ref_O.reset(static_cast<std::size_t>(batch) * num_heads_q * seq_len_qo * head_size_vo);

    initialize_block(block_Q, seed + 2023);
    initialize_block(block_K, seed + 2022);
    initialize_block(block_V, seed + 2021);

    return shape;
  }

  // Note that the GemmUniversalAdapter currently doesn't support flash attention, which is why this
  // secondary `run` function is required to launch the kernel.
  static void run(typename FMHAKernel::Params params)
  {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    // Launch parameters depend on whether SYCL compiler supports work-group scratch memory extension
    compat::experimental::launch_properties launch_props {
      syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
      syclex::sub_group_size<cute::intel::sg_size>,
      intelex::grf_size<256>
    };
    compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
    auto event = compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel>(policy, params);

    EventManager::getInstance().addEvent(event);
  }

  cutlass::Status run(const Options &options, const cutlass::KernelHardwareInfo &hw_info) {

    ProblemShapeType shape = initialize(options);

    typename FMHAKernel::Arguments arguments{
      {
        shape,
        block_Q.get(), stride_Q,
        block_K.get(), stride_K,
        block_V.get(), stride_V,
        block_O.get(), stride_O
      },
      {options.softmax_scale},
      {},
      hw_info
    };

    // Define device-global scratch memory
    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (!FMHAKernel::can_implement(arguments)) {
      std::cout << "Invalid Problem Size: " << options.batch << 'x' << options.num_heads_q << 'x' <<
        options.seq_len_qo << 'x' << options.seq_len_kv << 'x' << options.head_size_qk << 'x'  << options.head_size_vo
        << (options.is_causal ? "xCausal" : "xNonCausal") << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Initialize the workspace
    CUTLASS_CHECK(FMHAKernel::initialize_workspace(arguments, workspace.get()));

    // Convert host-side arguments to device-side arguments to be passed to the kernel
    auto params = FMHAKernel::to_underlying_arguments(arguments, workspace.get());

    // Run the GEMM
    run(params);

    compat::wait();

    // Verify that the result is correct
    bool passed = verify(shape, options.is_causal);
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
    // when seq_len_qo is not equal to seq_len_kv we use bottom up approach for the masking.
      // Following changes will adjust the effective_seq_len_kv when masking applied for such cases
      auto offset = cute::min(options.seq_len_qo, options.seq_len_kv);
      auto discard_seq_coord = options.seq_len_qo - offset;
      auto full_tile_offset = options.seq_len_kv - offset;
      // offset + 1 is going to be ceil_div
      auto effective_seq_len_kv = options.is_causal ? full_tile_offset + ((offset + 1) / 2.0): options.seq_len_kv;
      auto effective_seq_len_qo = options.is_causal ? options.seq_len_qo - discard_seq_coord : options.seq_len_qo;
      double cute_time = timer.seconds() / options.iterations;
      double flops_qk = 2.0 * options.batch * options.num_heads_q * effective_seq_len_qo * effective_seq_len_kv * options.head_size_qk;
      double flops_pv = 2.0 *  options.batch * options.num_heads_q * effective_seq_len_qo * options.head_size_vo * effective_seq_len_kv;
      double tflops = ((flops_qk + flops_pv) * 1e-12) / cute_time;
      double gbps_qk =  options.batch * (sizeof(ElementQ) * options.num_heads_q * effective_seq_len_qo * options.head_size_qk +
                                         sizeof(ElementK) * options.num_heads_kv * effective_seq_len_kv * options.head_size_qk);
      double gbps_pv = sizeof(ElementV) * options.batch * options.num_heads_kv * effective_seq_len_kv * options.head_size_vo +
                     sizeof(ElementO) * options.batch * options.num_heads_q * effective_seq_len_qo * options.head_size_vo;
      double gbps = ((gbps_qk + gbps_pv)  * 1e-9) / (cute_time);
      std::cout << "Batch: " << options.batch << "\tNumHeads_q: " << options.num_heads_q  << "\tNumHeads_kv: " << options.num_heads_kv  << "\tSeq Length QO: " << options.seq_len_qo
                << "\tSeq Length KV: " << options.seq_len_kv << "\tHead Size QK: " << options.head_size_qk << "\tHead Size VO: " << options.head_size_vo
                << "\tCausal Mask: " << (options.is_causal ? "true" : "false") << "\tVariable Sequence Length: " << (options.varlen ? "true" : "false")
                << "\t Scheduler: " << options.scheduler;
      printf("\nPerformance:   %4.3f  GB/s,    %4.3f  TFlop/s,   %6.4f  ms\n\n", gbps, tflops, cute_time * 1000);
    }

    return cutlass::Status::kSuccess;
  }
};

template <bool Causal,
          typename TileShapeQK,
          typename TileShapePV,
          typename TileShapeOutput,
          typename SubgroupLayoutQK,
          typename SubgroupLayoutPV_,      /* void -> default */
          int PipelineStages,
          typename ElementQ = bfloat16_t,
          typename ElementK = bfloat16_t,
          typename ElementV = bfloat16_t,
          typename ElementO = float,
          typename MMAOperation_ = void,    /* void -> default */
          typename StrideQ = Stride<int, _1, int, int>,
          typename StrideK = Stride<int, _1, int, int>,
          typename StrideV = Stride<_1, int, int, int>,
          typename StrideO = Stride<int, _1, int, int>,
          typename GmemTiledCopyQ = void,   /* void -> default block 2D */
          typename GmemTiledCopyK = void,
          typename GmemTiledCopyV = void,
          typename GmemTiledCopyO = void>
struct FMHAConfig {

  static constexpr int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation = cute::conditional_t<is_void_v<MMAOperation_>,
                                           XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>,
                                           MMAOperation_>;
  using SubgroupLayoutPV = cute::conditional_t<is_void_v<SubgroupLayoutPV_>,
                                               decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{})),
                                               SubgroupLayoutPV_>;

  template <bool isVarLen, class Scheduler>
  static int run(const Options &options) {
    //
    // Run examples
    //

    // The KernelHardwareInfo struct holds the number of EUs on the GPU with a given device ID. This
    // information is used by the underlying kernel.
    cutlass::KernelHardwareInfo hw_info;

    using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape;

    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;

    static_assert(get<0>(TileShapeOutput{}) == get<0>(TileShapePV{}),
        "Output tile and P*V tile have different sizes in Q dimension");
    constexpr int VTiles = get<1>(TileShapeOutput{}) / get<1>(TileShapePV{});

    auto make_dummy_tensor = [&](auto val, auto stride) {
      return make_tensor(make_gmem_ptr(&val),
                         make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
    };

    using TensorQ = decltype(make_dummy_tensor(ElementQ{}, StrideQ{}));
    using TensorK = decltype(make_dummy_tensor(ElementK{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementV{}, StrideV{}));
    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideO{}));

    // Mainloop
    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        MainloopDispatchPolicy, Causal,
        TiledMMAQK, TiledMMAPV, VTiles,
        TensorQ, TensorK, TensorV,
        GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV
    >;

    // Epilogue
    using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogue<
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        GmemTiledCopyO
    >;

    using FMHAKernel = cutlass::fmha::kernel::XeFMHAFwdKernel<
        ProblemShapeType, CollectiveMainloop, CollectiveEpilogue, Scheduler
    >;

    ExampleRunner<FMHAKernel> runner;

    CUTLASS_CHECK(runner.run(options, hw_info));
    return 0;
  }

  static int run(const Options &options) {
    return run<false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
  }
};
