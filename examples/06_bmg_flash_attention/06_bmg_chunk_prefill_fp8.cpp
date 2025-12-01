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
    \brief fp8 Chunk Prefill for Intel BMG

    This example constructs and executes a FP8 Flash Attention Chunk Prefill on Intel BMG. The
    definition of the GEMM, options etc for this example are defined in the associated
    bmg_flash_chunk_prefill_runner.hpp header file.

    See https://arxiv.org/pdf/2307.08691 for details of Flash Attention V2 algorithm

    To build & run this example (from your build dir):

      $ ninja 06_bmg_chunk_prefill_fp8_hdim128
      $ ./examples/06_bmg_flash_attention/06_bmg_chunk_prefill_fp8_hdim128

    Call with `--help` for information about available options
*/

#include "bmg_flash_chunk_prefill_runner.hpp"

int main(int argc, const char **argv) {
  //
  // Parse options
  //

  Options options;

  // Pre-parse command line to get batch and head counts for sizing the default vectors.
  cutlass::CommandLine cmd(argc, argv);
  int batch = 32;
  int num_heads_q = 16;
  int num_heads_kv = 0;
  cmd.get_cmd_line_argument("batch", batch, 32);
  cmd.get_cmd_line_argument("num_heads_q", num_heads_q, 16);
  cmd.get_cmd_line_argument("num_heads_kv", num_heads_kv, num_heads_q);

  // Set default scale values for this specific FP8 test using the correct 2D vector shapes.
  options.q_scale.assign(batch, std::vector<float>(num_heads_q, 1.5f));
  options.k_scale.assign(batch, std::vector<float>(num_heads_kv, 2.5f));
  options.v_scale.assign(batch, std::vector<float>(num_heads_kv, 1.9f));

  // Now, parse all arguments. The defaults we just set will be preserved unless
  // the user has provided their own scale values via the command line.
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

#if !defined(HEAD_DIM)
  std::cerr << "HEAD_DIM must be defined" << std::endl;
  return -1;
#endif
  if (options.head_size_vo != HEAD_DIM) {
    std::cerr << "head_size_vo must be " << HEAD_DIM << ", but got " << options.head_size_vo << std::endl;
    return -1;
  }

  // =================================================================================================
  // FP8 Type Definitions
  // =================================================================================================
  using ElementInputQ = cutlass::float_e5m2_t;     // <- data type of elements in input matrix A
  using ElementInputKV = cutlass::float_e5m2_t;    // <- data type of elements in input matrix B
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using GmemTiledCopyQ = XE_2D_U8x8x32_LD_N;
  using GmemTiledCopyK = XE_2D_U8x16x16_LD_T; // _T designates a transposed block load operation
  using GmemTiledCopyV = XE_2D_U8x32x32_LD_V;

  constexpr int PipelineStages = 2;

  // =================================================================================================
  // Tile Shape Definitions
  // =================================================================================================
#if HEAD_DIM == 64
  using ShapeQK = Shape<_128, _64, _64>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutPut = Shape<_128, _64, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>;
#elif HEAD_DIM == 96
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutPut = Shape<_128, _96, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>;
#elif HEAD_DIM == 128
  using ShapeQK = Shape<_128, _64, _64>;
    using ShapePV = Shape<_128, _32, _64>;
    using ShapeOutPut = Shape<_128, _128, _64>;
    using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>;
#elif HEAD_DIM == 192
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutPut = Shape<_256, _192, _64>;
  using SubgroupLayout = Layout<Shape<_32, _1, _1>, Stride<_1, _1, _1>>;
#endif

  // =================================================================================================
  // Kernel Launch
  // =================================================================================================
  if (options.is_causal) {
    FMHAConfig<true, false, ShapeQK, ShapePV, ShapeOutPut, SubgroupLayout, PipelineStages,
                                          ElementInputQ, ElementInputKV, MMAOperation,
                                          GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV>::run(options);
  } else if (options.is_local_mask) {
    FMHAConfig<false, true, ShapeQK, ShapePV, ShapeOutPut, SubgroupLayout, PipelineStages,
                                          ElementInputQ, ElementInputKV, MMAOperation,
                                          GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV>::run(options);
  } else {
    FMHAConfig<false, false, ShapeQK, ShapePV, ShapeOutPut, SubgroupLayout, PipelineStages,
                                          ElementInputQ, ElementInputKV, MMAOperation,
                                          GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV>::run(options);
  }
}
