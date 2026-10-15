/* Copyright (C) 2026 Intel Corporation, All rights reserved.
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

// Shared helpers for the Xe2 e4m3 -> half reorder tests (reorder_e4m3_half.cpp: default path,
// reorder_e4m3_half_unscaled.cpp: CUTE_XE_E4M3_HALF_UNSCALED path). Include after the CuTe headers.
//
// Each test runs one Xe_Reorder<UU|VV, float_e4m3_t, half_t> atom per 64-byte input register over all 256
// e4m3 codes and compares every output bit pattern against a host reference.

#pragma once

#include <cute/tensor.hpp>
#include <cute/arch/reorder_xe.hpp>
#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>

#include "cutlass_unit_test.h"

namespace e4m3_half_test {

using namespace cute;
using namespace compat::experimental;

// A sub-group register of N-component vectors holds component c of work-item i at element c * 16 + i.
// One uchar4 per work-item = one 64-byte input register; one ushort4 = two 32-word output registers.
constexpr int kSg = intel::sg_size;          // 16
constexpr int kBytesPerReg = 4 * kSg;        // 64
constexpr int kRegs = 256 / kBytesPerReg;    // 4 registers cover all 256 codes

template <ReorderKind Kind>
void reorder_kernel(uint8_t const* src, uint16_t* dst) {
  const int tid = ThreadIdxX();
  for (int r = 0; r < kRegs; ++r) {
    intel::uchar4 s;
    intel::ushort4 d;
    for (int c = 0; c < 4; ++c) {
      s[c] = src[r * kBytesPerReg + c * kSg + tid];
    }
    Xe_Reorder<Kind, cutlass::float_e4m3_t, cutlass::half_t>::reorder(s, d);
    for (int c = 0; c < 4; ++c) {
      dst[r * kBytesPerReg + c * kSg + tid] = d[c];
    }
  }
}

// Source byte (within the 64-byte register) feeding output word w (0..63) of the register pair.
//   UU: <1;1,0> regions, word w <- byte w.
//   VV: <4;2,1> regions from offsets 0 and 2: words 0..31 take bytes 0,1,4,5,8,9,...;
//       words 32..63 take bytes 2,3,6,7,...
template <ReorderKind Kind>
int source_byte(int w) {
  if constexpr (Kind == ReorderKind::UU) {
    return w;
  } else {
    const int h = w / 32, v = w % 32;
    return (v / 2) * 4 + 2 * h + (v % 2);
  }
}

inline uint16_t half_bits(float f) {
  cutlass::half_t h(f);
  uint16_t b;
  std::memcpy(&b, &h, sizeof(b));
  return b;
}

inline bool half_is_nan(uint16_t b) { return (b & 0x7C00) == 0x7C00 && (b & 0x03FF) != 0; }

// Run the atom on the device for all 256 codes and check every output against the host reference.
//   Unscaled == false: out == half(e4m3), NaN codes (0x7F, 0xFF) -> NaN (default path).
//   Unscaled == true:  out == half(e4m3 * 2^-8) exactly, NaN codes -> +-480 * 2^-8 (opt-in path).
template <ReorderKind Kind, bool Unscaled, class KernelName>
void run_all_codes() {
  cutlass::host_vector<uint8_t> host_src(256);
  for (int i = 0; i < 256; ++i) host_src[i] = static_cast<uint8_t>(i);
  cutlass::device_vector<uint8_t> device_src = host_src;
  cutlass::device_vector<uint16_t> device_dst(256);

  launch<reorder_kernel<Kind>, KernelName>(
      launch_policy{compat::dim3(1), compat::dim3(kSg),
                    kernel_properties{sycl_exp::sub_group_size<kSg>}},
      device_src.data(), device_dst.data());
  compat::wait_and_throw();
  cutlass::host_vector<uint16_t> host_dst = device_dst;

  int checked = 0;
  for (int r = 0; r < kRegs; ++r) {
    for (int w = 0; w < kBytesPerReg; ++w) {
      const uint8_t code = host_src[r * kBytesPerReg + source_byte<Kind>(w)];
      const uint16_t got = host_dst[r * kBytesPerReg + w];
      const bool nan_code = (code & 0x7F) == 0x7F;
      float ref = static_cast<float>(cutlass::float_e4m3_t::bitcast(code));
      if (Unscaled) {
        if (nan_code) ref = (code & 0x80) ? -480.0f : 480.0f;
        ref = std::ldexp(ref, -8);
        EXPECT_EQ(got, half_bits(ref)) << "code 0x" << std::hex << int(code) << " word " << std::dec << w;
      } else if (nan_code) {
        EXPECT_TRUE(half_is_nan(got)) << "code 0x" << std::hex << int(code) << " -> 0x" << got;
      } else {
        EXPECT_EQ(got, half_bits(ref)) << "code 0x" << std::hex << int(code) << " word " << std::dec << w;
      }
      ++checked;
    }
  }
  EXPECT_EQ(checked, 256);
}

} // namespace e4m3_half_test
