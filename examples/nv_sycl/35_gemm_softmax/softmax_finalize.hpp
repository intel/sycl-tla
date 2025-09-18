/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
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
  \brief Kernel performing a final calculation of softmax
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reduction {
namespace kernel {

template <
  typename ElementInput_,
  typename StrideInput_,
  typename ElementPartial_,
  typename StridePartial_,
  typename ElementOutput_,
  typename StrideOutput_
>
class SoftmaxFinalize {
public:

  using ElementInput = ElementInput_;
  using StrideInput = StrideInput_;
  using ElementPartial = ElementPartial_;
  using StridePartial = StridePartial_;
  using ElementOutput = ElementOutput_;
  using StrideOutput = StrideOutput_;

  //
  // Arguments
  //

  struct Arguments {
    int                            M; // dimension M of input, output and partially reduced tensors
    int                        dataN; // dimension N of the input and output
    int                     partialN; // dimension N of the partially reduced tensors
    int                  batch_count; // batch count
    StrideInput               dInput; // stride of the input
    StridePartial           dPartial; // stride of the partially reduced tensors
    StrideOutput             dOutput; // stride of the output
    ElementInput*             ptr_in; // pointer to start of input data
    ElementPartial*  ptr_partial_max; // pointer to start of partially reduced max data
    ElementPartial*  ptr_partial_sum; // pointer to start of partially reduced sum data
    ElementOutput*           ptr_out; // pointer to start of output data
  };

  struct SharedStorage {
    cute::array_aligned<ElementPartial, MaxNumThreadsPerBlock> s_mem;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  //
  // Params struct
  //

  struct Params {
    Arguments args;

    //
    // Methods
    //
    Params() { }

    Params(Arguments const &args_): args(args_) { }
  };

public:

  CUTLASS_DEVICE
  SoftmaxFinalize() { }

  CUTLASS_DEVICE
  void operator()(Params const &params, char* shared_storage) {
    apply(params, shared_storage);
  }

private:

  CUTLASS_DEVICE
  void apply(Params const &params, char* shared_storage) {
    using ConvertInput = cutlass::NumericConverter<ElementInput, ElementPartial>;
    using ConvertNormOutput = cutlass::NumericConverter<ElementPartial, ElementOutput>;

    const int idx_x = ThreadIdxX();
    const int m = idx_x + BlockDimX() * BlockIdxX();
    const int idx_y = ThreadIdxY();
    const int y_size = BlockDimY();
    const int batch_id = BlockIdxY();

    if (m >= params.args.M) {
      return;
    }

    // Represent the full tensors
    auto IOTensorShape = make_shape(params.args.M, params.args.dataN, params.args.batch_count);
    auto PartialTensorShape = make_shape(params.args.M, params.args.partialN, params.args.batch_count);
    Tensor mPartialMax = make_tensor(make_gmem_ptr(params.args.ptr_partial_max), PartialTensorShape, params.args.dPartial);
    Tensor mPartialSum = make_tensor(make_gmem_ptr(params.args.ptr_partial_sum), PartialTensorShape, params.args.dPartial);
    Tensor mOut = make_tensor(make_gmem_ptr(params.args.ptr_out), IOTensorShape, params.args.dOutput);
    Tensor mIn = make_tensor(make_gmem_ptr(params.args.ptr_in), IOTensorShape, params.args.dInput);

    //Represent the shared tensor
    Tensor sPartial = make_tensor(make_smem_ptr(reinterpret_cast<ElementPartial*>(shared_storage)), 
                                  make_layout(make_shape(NumThreadsPerWarp, MaxNumThreadsPerBlock / NumThreadsPerWarp)));

    ElementPartial max_val = std::numeric_limits<ElementPartial>::lowest();
    for (int partial_n = idx_y; partial_n < params.args.partialN; partial_n += y_size){
        ElementPartial partial_max = mPartialMax(m, partial_n, batch_id);
        max_val = cutlass::fast_max(max_val, partial_max);
    }
    sPartial(idx_x, idx_y) = max_val;
    syncthreads();
    // tree-reduction could be better, although it does not seem to be a bottleneck
    for (int idx_y2 = 0; idx_y2 < y_size; idx_y2++){
        ElementPartial partial_max = sPartial(idx_x, idx_y2);
        max_val = cutlass::fast_max(max_val, partial_max);
    }
    
    ElementPartial sum_val = 0;
    for (int partial_n = idx_y; partial_n < params.args.partialN; partial_n += y_size){
        ElementPartial partial_max = mPartialMax(m, partial_n, batch_id);
        ElementPartial partial_sum = mPartialSum(m, partial_n, batch_id);
        sum_val += partial_sum * cutlass::fast_exp(partial_max - max_val);
    }
    syncthreads();
    sPartial(idx_x, idx_y) = sum_val;
    syncthreads();
    sum_val = 0;
    // tree-reduction could be better, although it does not seem to be a bottleneck
    for(int idx_y2 = 0; idx_y2 < y_size; idx_y2++){
        ElementPartial partial_sum = sPartial(idx_x, idx_y2);
        sum_val += partial_sum;
    }

    ElementPartial norm = 1 / sum_val;

    for (int n = idx_y * 2; n < params.args.dataN; n += y_size * 2){
      auto inVal = mIn(m, n, batch_id);
      auto inVal2 = mIn(m, n+1, batch_id);
      mOut(m, n, batch_id) = cutlass::fast_exp(inVal - max_val) * norm;
      mOut(m, n+1, batch_id) = cutlass::fast_exp(inVal2 - max_val) * norm;
    }
    if (params.args.dataN % 2 == 1){
      int n = params.args.dataN - 1;
      auto inVal = mIn(m, n, batch_id);
      mOut(m, n, batch_id) = cutlass::fast_exp(inVal - max_val) * norm;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace reduction
} // namespace cutlass
