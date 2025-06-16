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

#pragma once

#include <benchmarks_h64_fp32_fp32_512.cpp>
#include <benchmarks_h64_fp32_fp32_1024.cpp>
#include <benchmarks_h96_fp32_fp32_512.cpp>
#include <benchmarks_h96_fp32_fp32_1024.cpp>
#include <benchmarks_h128_fp32_fp32_512.cpp>
#include <benchmarks_h128_fp32_fp32_1024.cpp>
#include <benchmarks_h192_fp32_fp32_512.cpp>
#include <benchmarks_h192_fp32_fp32_1024.cpp>

#include <benchmarks_h64_fp32_u16_512.cpp>
#include <benchmarks_h64_fp32_u16_1024.cpp>
#include <benchmarks_h96_fp32_u16_512.cpp>
#include <benchmarks_h96_fp32_u16_1024.cpp>
#include <benchmarks_h128_fp32_u16_512.cpp>
#include <benchmarks_h128_fp32_u16_1024.cpp>
#include <benchmarks_h192_fp32_u16_512.cpp>
#include <benchmarks_h192_fp32_u16_1024.cpp>

static void register_flash_attention_decode_benchmarks() {
  register_flash_attention_decode_benchmarks_h64_fp32_fp32_512();
  register_flash_attention_decode_benchmarks_h96_fp32_fp32_512();
  register_flash_attention_decode_benchmarks_h128_fp32_fp32_512();
  register_flash_attention_decode_benchmarks_h192_fp32_fp32_512();
  register_flash_attention_decode_benchmarks_h64_fp32_fp32_1024();
  register_flash_attention_decode_benchmarks_h96_fp32_fp32_1024();
  register_flash_attention_decode_benchmarks_h128_fp32_fp32_1024();
  register_flash_attention_decode_benchmarks_h192_fp32_fp32_1024();

  register_flash_attention_decode_benchmarks_h64_fp32_u16_512();
  register_flash_attention_decode_benchmarks_h96_fp32_u16_512();
  register_flash_attention_decode_benchmarks_h128_fp32_u16_512();
  register_flash_attention_decode_benchmarks_h192_fp32_u16_512();
  register_flash_attention_decode_benchmarks_h64_fp32_u16_1024();
  register_flash_attention_decode_benchmarks_h96_fp32_u16_1024();
  register_flash_attention_decode_benchmarks_h128_fp32_u16_1024();
  register_flash_attention_decode_benchmarks_h192_fp32_u16_1024();
}
