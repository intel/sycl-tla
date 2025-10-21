/***************************************************************************************************
 * Copyright (c) 2025 Intel Corporation. All rights reserved.
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

namespace cutlass::flash_attention::collective {
using namespace cute;

template <typename Tensor,
          typename TensorCos, typename TensorSin, typename TensorOut>
CUTLASS_DEVICE void apply_rope_interleaved_gmem(
    int thread_idx,
    Tensor const &srcTensor,
    TensorCos const &gCos,
    TensorSin const &gSin, TensorOut &destTensor) {
  if(thread_idx < size<0>(srcTensor)){
    for (int j = 0; j < size<1>(gCos); j+=2) {
        auto real = static_cast<float>(srcTensor[make_coord(thread_idx, j)]);
        auto imag = static_cast<float>(srcTensor[make_coord(thread_idx, j + 1)]);
        auto cos_val = static_cast<float>(gCos[make_coord(thread_idx, j)]);
        auto sin_val = static_cast<float>(gSin[make_coord(thread_idx, j)]);

        auto new_real = real * cos_val - imag * sin_val;
        auto new_imag = real * sin_val + imag * cos_val;

        destTensor[make_coord(thread_idx,j)] = static_cast<typename Tensor::value_type>(new_real);
        destTensor[make_coord(thread_idx,j + 1)] = static_cast<typename Tensor::value_type>(new_imag);
    }
  }
  syncthreads();
}
} // namespace cutlass::flash_attention::collective