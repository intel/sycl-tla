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

#pragma once

#include "cute/tensor.hpp"
#include "cute/util/sycl_vec.hpp"

namespace cute {

// Uniformize a value, in case the compiler cannot prove it is subgroup-uniform.
template <typename T>
CUTE_HOST_DEVICE
T
assert_uniform(T x) {
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  return group_broadcast(sg, x, 0);
}

// Set a value in a single work-item -- x[i] = val.
// WARNING: i _must_ be a compile-time constant.
//   No diagnostics/error will be issued by the compiler if it is not.
template <typename T>
CUTE_HOST_DEVICE void
set_wi_value(T &x, int i, T val)
{
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
  asm (
    "mov (M1_NM, 1) %0(0,%2)<1> %1(0,0)<1;1,0>"
    : "+rw"(x)
    : "rw.u"(val), "P"(i)
  );
#else
  int lane = sycl::ext::oneapi::this_work_item::get_sub_group().get_local_id()[0];
  if (lane == i)
    x = val;
#endif
}

// Set an element of a 1D SG-shared fragment x.
// WARNING: i _must_ be a compile-time constant.
//   No diagnostics/error will be issued by the compiler if it is not.
template <typename FragX>
CUTE_HOST_DEVICE void
set_single_value(FragX& x, int i, typename FragX::element_type val) {
  set_wi_value(x(i / intel::sg_size), i % intel::sg_size, val);
}

// Broadcast the element from a 1D SG-shared fragment x
//   corresponding to the Mode'th dimension of the logical coordinates of src(val).
template <int Mode, typename FragX, typename SGTensorSrc,
          __CUTE_REQUIRES(is_sg_tensor<SGTensorSrc>::value)>
CUTE_HOST_DEVICE
constexpr auto
broadcast(FragX const& x, SGTensorSrc const& src, int val)
{
  auto coord = src.tv_layout()(0, val);
  auto coord_i = get<Mode>(coord);

  constexpr auto TMode = rank(as_arithmetic_tuple(stride<0>(SGTensorSrc{}.tv_layout()))) - 1;
  if constexpr (TMode == Mode) {
    return x(coord_i / intel::sg_size);
  } else {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    return group_broadcast(sg, x(coord_i / intel::sg_size), coord_i % intel::sg_size);
  }
}

#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
#define DEFINE_HREDUCE16_FLOAT(op) \
  CUTE_DEVICE \
  float \
  hreduce16_float_ ## op(float x[16]) \
  { \
    float y; \
    asm ( \
      "{\n" \
      ".decl INTERLEAVE_2 v_type=P num_elts=16\n" \
      ".decl INTERLEAVE_4 v_type=P num_elts=16\n" \
      ".decl INTERLEAVE_8 v_type=P num_elts=16\n" \
      ".decl IN0 v_type=G type=UD num_elts=16 alias=<%1,0>\n" \
      ".decl IN1 v_type=G type=UD num_elts=16 alias=<%2,0>\n" \
      ".decl IN2 v_type=G type=UD num_elts=16 alias=<%3,0>\n" \
      ".decl IN3 v_type=G type=UD num_elts=16 alias=<%4,0>\n" \
      ".decl IN4 v_type=G type=UD num_elts=16 alias=<%5,0>\n" \
      ".decl IN5 v_type=G type=UD num_elts=16 alias=<%6,0>\n" \
      ".decl IN6 v_type=G type=UD num_elts=16 alias=<%7,0>\n" \
      ".decl IN7 v_type=G type=UD num_elts=16 alias=<%8,0>\n" \
      ".decl IN8 v_type=G type=UD num_elts=16 alias=<%9,0>\n" \
      ".decl IN9 v_type=G type=UD num_elts=16 alias=<%10,0>\n" \
      ".decl IN10 v_type=G type=UD num_elts=16 alias=<%11,0>\n" \
      ".decl IN11 v_type=G type=UD num_elts=16 alias=<%12,0>\n" \
      ".decl IN12 v_type=G type=UD num_elts=16 alias=<%13,0>\n" \
      ".decl IN13 v_type=G type=UD num_elts=16 alias=<%14,0>\n" \
      ".decl IN14 v_type=G type=UD num_elts=16 alias=<%15,0>\n" \
      ".decl IN15 v_type=G type=UD num_elts=16 alias=<%16,0>\n" \
      ".decl RA0 v_type=G type=UD num_elts=32 align=64\n" \
      ".decl RA2 v_type=G type=UD num_elts=32 align=64\n" \
      ".decl RA4 v_type=G type=UD num_elts=32 align=64\n" \
      ".decl RA6 v_type=G type=UD num_elts=32 align=64\n" \
      ".decl RA8 v_type=G type=UD num_elts=32 align=64\n" \
      ".decl RA10 v_type=G type=UD num_elts=32 align=64\n" \
      ".decl RA12 v_type=G type=UD num_elts=32 align=64\n" \
      ".decl RA14 v_type=G type=UD num_elts=32 align=64\n" \
      ".decl RF0 v_type=G type=F num_elts=16 alias=<RA0,0>\n" \
      ".decl RF1 v_type=G type=F num_elts=16 alias=<RA0,64>\n" \
      ".decl RF2 v_type=G type=F num_elts=16 alias=<RA2,0>\n" \
      ".decl RF3 v_type=G type=F num_elts=16 alias=<RA2,64>\n" \
      ".decl RF4 v_type=G type=F num_elts=16 alias=<RA4,0>\n" \
      ".decl RF5 v_type=G type=F num_elts=16 alias=<RA4,64>\n" \
      ".decl RF6 v_type=G type=F num_elts=16 alias=<RA6,0>\n" \
      ".decl RF7 v_type=G type=F num_elts=16 alias=<RA6,64>\n" \
      ".decl RF8 v_type=G type=F num_elts=16 alias=<RA8,0>\n" \
      ".decl RF9 v_type=G type=F num_elts=16 alias=<RA8,64>\n" \
      ".decl RF10 v_type=G type=F num_elts=16 alias=<RA10,0>\n" \
      ".decl RF11 v_type=G type=F num_elts=16 alias=<RA10,64>\n" \
      ".decl RF12 v_type=G type=F num_elts=16 alias=<RA12,0>\n" \
      ".decl RF13 v_type=G type=F num_elts=16 alias=<RA12,64>\n" \
      ".decl RF14 v_type=G type=F num_elts=16 alias=<RA14,0>\n" \
      ".decl RF15 v_type=G type=F num_elts=16 alias=<RA14,64>\n" \
      "setp (M1_NM,16) INTERLEAVE_2 0x5555:uw\n" \
      "setp (M1_NM,16) INTERLEAVE_4 0x3333:uw\n" \
      "setp (M1_NM,16) INTERLEAVE_8 0x0F0F:uw\n" \
      /* Round 1: interleave 2n with 2n+1 */ \
      "(!INTERLEAVE_2) sel (M1_NM,16) RA0(0,0)<1>  IN1(0,0)<2;2,0>  IN0(0,0)<1;1,0>\n" \
      " (INTERLEAVE_2) sel (M1_NM,16) RA0(1,0)<1>  IN0(0,1)<2;2,0>  IN1(0,0)<1;1,0>\n" \
      "(!INTERLEAVE_2) sel (M1_NM,16) RA2(0,0)<1>  IN3(0,0)<2;2,0>  IN2(0,0)<1;1,0>\n" \
      " (INTERLEAVE_2) sel (M1_NM,16) RA2(1,0)<1>  IN2(0,1)<2;2,0>  IN3(0,0)<1;1,0>\n" \
      "(!INTERLEAVE_2) sel (M1_NM,16) RA4(0,0)<1>  IN5(0,0)<2;2,0>  IN4(0,0)<1;1,0>\n" \
      " (INTERLEAVE_2) sel (M1_NM,16) RA4(1,0)<1>  IN4(0,1)<2;2,0>  IN5(0,0)<1;1,0>\n" \
      "(!INTERLEAVE_2) sel (M1_NM,16) RA6(0,0)<1>  IN7(0,0)<2;2,0>  IN6(0,0)<1;1,0>\n" \
      " (INTERLEAVE_2) sel (M1_NM,16) RA6(1,0)<1>  IN6(0,1)<2;2,0>  IN7(0,0)<1;1,0>\n" \
      "(!INTERLEAVE_2) sel (M1_NM,16) RA8(0,0)<1>  IN9(0,0)<2;2,0>  IN8(0,0)<1;1,0>\n" \
      " (INTERLEAVE_2) sel (M1_NM,16) RA8(1,0)<1>  IN8(0,1)<2;2,0>  IN9(0,0)<1;1,0>\n" \
      "(!INTERLEAVE_2) sel (M1_NM,16) RA10(0,0)<1> IN11(0,0)<2;2,0> IN10(0,0)<1;1,0>\n" \
      " (INTERLEAVE_2) sel (M1_NM,16) RA10(1,0)<1> IN10(0,1)<2;2,0> IN11(0,0)<1;1,0>\n" \
      "(!INTERLEAVE_2) sel (M1_NM,16) RA12(0,0)<1> IN13(0,0)<2;2,0> IN12(0,0)<1;1,0>\n" \
      " (INTERLEAVE_2) sel (M1_NM,16) RA12(1,0)<1> IN12(0,1)<2;2,0> IN13(0,0)<1;1,0>\n" \
      "(!INTERLEAVE_2) sel (M1_NM,16) RA14(0,0)<1> IN15(0,0)<2;2,0> IN14(0,0)<1;1,0>\n" \
      " (INTERLEAVE_2) sel (M1_NM,16) RA14(1,0)<1> IN14(0,1)<2;2,0> IN15(0,0)<1;1,0>\n" \
      /* Reduce */ \
      #op " (M1_NM,16) RF0(0,0)<1> RF0(0,0)<1;1,0> RF1(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) RF3(0,0)<1> RF2(0,0)<1;1,0> RF3(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) RF4(0,0)<1> RF4(0,0)<1;1,0> RF5(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) RF7(0,0)<1> RF6(0,0)<1;1,0> RF7(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) RF8(0,0)<1> RF8(0,0)<1;1,0> RF9(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) RF11(0,0)<1> RF10(0,0)<1;1,0> RF11(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) RF12(0,0)<1> RF12(0,0)<1;1,0> RF13(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) RF15(0,0)<1> RF14(0,0)<1;1,0> RF15(0,0)<1;1,0>\n" \
      /* Round 2: interleave 4n+{0,1} with 4n+{2,3} */ \
      "(!INTERLEAVE_4) sel (M1_NM,16) RA0(1,0)<1>  RA2(0,14)<1;1,0>  RA0(0,0)<1;1,0>\n" \
      " (INTERLEAVE_4) sel (M1_NM,16) RA0(0,0)<1>  RA0(0,2)<1;1,0>   RA2(1,0)<1;1,0>\n" \
      "(!INTERLEAVE_4) sel (M1_NM,16) RA4(1,0)<1>  RA6(0,14)<1;1,0>  RA4(0,0)<1;1,0>\n" \
      " (INTERLEAVE_4) sel (M1_NM,16) RA4(0,0)<1>  RA4(0,2)<1;1,0>   RA6(1,0)<1;1,0>\n" \
      "(!INTERLEAVE_4) sel (M1_NM,16) RA8(1,0)<1>  RA10(0,14)<1;1,0> RA8(0,0)<1;1,0>\n" \
      " (INTERLEAVE_4) sel (M1_NM,16) RA8(0,0)<1>  RA8(0,2)<1;1,0>   RA10(1,0)<1;1,0>\n" \
      "(!INTERLEAVE_4) sel (M1_NM,16) RA12(1,0)<1> RA14(0,14)<1;1,0> RA12(0,0)<1;1,0>\n" \
      " (INTERLEAVE_4) sel (M1_NM,16) RA12(0,0)<1> RA12(0,2)<1;1,0>  RA14(1,0)<1;1,0>\n" \
      /* Reduce */ \
      #op " (M1_NM,16) RF0(0,0)<1>  RF0(0,0)<1;1,0>  RF1(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) RF5(0,0)<1>  RF4(0,0)<1;1,0>  RF5(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) RF8(0,0)<1>  RF8(0,0)<1;1,0>  RF9(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) RF13(0,0)<1> RF12(0,0)<1;1,0> RF13(0,0)<1;1,0>\n" \
      /* Round 3: interleave 8n+{0,1,2,3} with 8n+{4,5,6,7} */ \
      "(!INTERLEAVE_8) sel (M1_NM,16) RA0(1,0)<1> RA4(0,12)<1;1,0>  RA0(0,0)<1;1,0>\n" \
      " (INTERLEAVE_8) sel (M1_NM,16) RA0(0,0)<1> RA0(0,4)<1;1,0>   RA4(1,0)<1;1,0>\n" \
      "(!INTERLEAVE_8) sel (M1_NM,16) RA8(1,0)<1> RA12(0,12)<1;1,0> RA8(0,0)<1;1,0>\n" \
      " (INTERLEAVE_8) sel (M1_NM,16) RA8(0,0)<1> RA8(0,4)<1;1,0>   RA12(1,0)<1;1,0>\n" \
      /* Reduce */ \
      #op " (M1_NM,16) RF0(0,0)<1> RF0(0,0)<1;1,0> RF1(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) RF8(0,0)<1> RF8(0,0)<1;1,0> RF9(0,0)<1;1,0>\n" \
      /* Round 4: final interleave */ \
      "mov (M1_NM, 8) RA0(1,0)<1> RA0(0,8)<1;1,0>\n" \
      "mov (M1_NM, 8) RA8(1,8)<1> RA8(0,0)<1;1,0>\n" \
      /* Reduce */ \
      #op " (M1_NM,8) %0(0,0)<1> RF0(0,0)<1;1,0> RF1(0,0)<1;1,0>\n" \
      #op " (M1_NM,8) %0(0,8)<1> RF8(0,8)<1;1,0> RF9(0,8)<1;1,0>\n" \
      "}\n" \
      : "=rw"(y) \
      : "rw"(x[0]), "rw"(x[1]), "rw"(x[2]),  "rw"(x[3]),  "rw"(x[4]),  "rw"(x[5]),  "rw"(x[6]),  "rw"(x[7]), \
        "rw"(x[8]), "rw"(x[9]), "rw"(x[10]), "rw"(x[11]), "rw"(x[12]), "rw"(x[13]), "rw"(x[14]), "rw"(x[15]) \
    ); \
    return y; \
  }

#define DEFINE_HREDUCE8_FLOAT(op) \
  CUTE_DEVICE \
  float \
  hreduce8_float_ ## op(float x[8]) \
  { \
    float y; \
    asm ( \
      "{\n" \
      ".decl INTERLEAVE_2 v_type=P num_elts=16\n" \
      ".decl INTERLEAVE_4 v_type=P num_elts=16\n" \
      ".decl INTERLEAVE_8 v_type=P num_elts=16\n" \
      ".decl IN0 v_type=G type=UD num_elts=16 alias=<%1,0>\n" \
      ".decl IN1 v_type=G type=UD num_elts=16 alias=<%2,0>\n" \
      ".decl IN2 v_type=G type=UD num_elts=16 alias=<%3,0>\n" \
      ".decl IN3 v_type=G type=UD num_elts=16 alias=<%4,0>\n" \
      ".decl IN4 v_type=G type=UD num_elts=16 alias=<%5,0>\n" \
      ".decl IN5 v_type=G type=UD num_elts=16 alias=<%6,0>\n" \
      ".decl IN6 v_type=G type=UD num_elts=16 alias=<%7,0>\n" \
      ".decl IN7 v_type=G type=UD num_elts=16 alias=<%8,0>\n" \
      ".decl RA0 v_type=G type=UD num_elts=32 align=64\n" \
      ".decl RA2 v_type=G type=UD num_elts=32 align=64\n" \
      ".decl RA4 v_type=G type=UD num_elts=32 align=64\n" \
      ".decl RA6 v_type=G type=UD num_elts=32 align=64\n" \
      ".decl RF0 v_type=G type=F num_elts=16 alias=<RA0,0>\n" \
      ".decl RF1 v_type=G type=F num_elts=16 alias=<RA0,64>\n" \
      ".decl RF2 v_type=G type=F num_elts=16 alias=<RA2,0>\n" \
      ".decl RF3 v_type=G type=F num_elts=16 alias=<RA2,64>\n" \
      ".decl RF4 v_type=G type=F num_elts=16 alias=<RA4,0>\n" \
      ".decl RF5 v_type=G type=F num_elts=16 alias=<RA4,64>\n" \
      ".decl RF6 v_type=G type=F num_elts=16 alias=<RA6,0>\n" \
      ".decl RF7 v_type=G type=F num_elts=16 alias=<RA6,64>\n" \
      "setp (M1_NM,16) INTERLEAVE_2 0x5555:uw\n" \
      "setp (M1_NM,16) INTERLEAVE_4 0x3333:uw\n" \
      "setp (M1_NM,16) INTERLEAVE_8 0x0F0F:uw\n" \
      /* Round 1: interleave 2n with 2n+1 */ \
      "(!INTERLEAVE_2) sel (M1_NM,16) RA0(0,0)<1>  IN1(0,0)<2;2,0>  IN0(0,0)<1;1,0>\n" \
      " (INTERLEAVE_2) sel (M1_NM,16) RA0(1,0)<1>  IN0(0,1)<2;2,0>  IN1(0,0)<1;1,0>\n" \
      "(!INTERLEAVE_2) sel (M1_NM,16) RA2(0,0)<1>  IN3(0,0)<2;2,0>  IN2(0,0)<1;1,0>\n" \
      " (INTERLEAVE_2) sel (M1_NM,16) RA2(1,0)<1>  IN2(0,1)<2;2,0>  IN3(0,0)<1;1,0>\n" \
      "(!INTERLEAVE_2) sel (M1_NM,16) RA4(0,0)<1>  IN5(0,0)<2;2,0>  IN4(0,0)<1;1,0>\n" \
      " (INTERLEAVE_2) sel (M1_NM,16) RA4(1,0)<1>  IN4(0,1)<2;2,0>  IN5(0,0)<1;1,0>\n" \
      "(!INTERLEAVE_2) sel (M1_NM,16) RA6(0,0)<1>  IN7(0,0)<2;2,0>  IN6(0,0)<1;1,0>\n" \
      " (INTERLEAVE_2) sel (M1_NM,16) RA6(1,0)<1>  IN6(0,1)<2;2,0>  IN7(0,0)<1;1,0>\n" \
      /* Reduce */ \
      #op " (M1_NM,16) RF0(0,0)<1> RF0(0,0)<1;1,0> RF1(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) RF3(0,0)<1> RF2(0,0)<1;1,0> RF3(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) RF4(0,0)<1> RF4(0,0)<1;1,0> RF5(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) RF7(0,0)<1> RF6(0,0)<1;1,0> RF7(0,0)<1;1,0>\n" \
      /* Round 2: interleave 4n+{0,1} with 4n+{2,3} */ \
      "(!INTERLEAVE_4) sel (M1_NM,16) RA0(1,0)<1>  RA2(0,14)<1;1,0>  RA0(0,0)<1;1,0>\n" \
      " (INTERLEAVE_4) sel (M1_NM,16) RA0(0,0)<1>  RA0(0,2)<1;1,0>   RA2(1,0)<1;1,0>\n" \
      "(!INTERLEAVE_4) sel (M1_NM,16) RA4(1,0)<1>  RA6(0,14)<1;1,0>  RA4(0,0)<1;1,0>\n" \
      " (INTERLEAVE_4) sel (M1_NM,16) RA4(0,0)<1>  RA4(0,2)<1;1,0>   RA6(1,0)<1;1,0>\n" \
      /* Reduce */ \
      #op " (M1_NM,16) RF0(0,0)<1>  RF0(0,0)<1;1,0>  RF1(0,0)<1;1,0>\n" \
      #op " (M1_NM,16) RF5(0,0)<1>  RF4(0,0)<1;1,0>  RF5(0,0)<1;1,0>\n" \
      /* Round 3: interleave 8n+{0,1,2,3} with 8n+{4,5,6,7} */ \
      "(!INTERLEAVE_8) sel (M1_NM,16) RA0(1,0)<1> RA4(0,12)<1;1,0>  RA0(0,0)<1;1,0>\n" \
      " (INTERLEAVE_8) sel (M1_NM,16) RA0(0,0)<1> RA0(0,4)<1;1,0>   RA4(1,0)<1;1,0>\n" \
      /* Reduce */ \
      #op " (M1_NM,16) RF0(0,0)<1> RF0(0,0)<1;1,0> RF1(0,0)<1;1,0>\n" \
      /* Round 4: reduce top and bottom halves */ \
      #op " (M1_NM,8) %0(0,0)<1> RF0(0,0)<1;1,0> RF0(0,8)<1;1,0>\n" \
      "}\n" \
      : "=rw"(y) \
      : "rw"(x[0]), "rw"(x[1]), "rw"(x[2]),  "rw"(x[3]),  "rw"(x[4]),  "rw"(x[5]),  "rw"(x[6]),  "rw"(x[7]), \
        "rw"(x[8]), "rw"(x[9]), "rw"(x[10]), "rw"(x[11]), "rw"(x[12]), "rw"(x[13]), "rw"(x[14]), "rw"(x[15]) \
    ); \
    return y; \
  }
#else
#define DEFINE_HREDUCE16_FLOAT(op) \
  CUTE_DEVICE float hreduce16_float_ ## op(float x[16]) { return 0.f; }
#define DEFINE_HREDUCE8_FLOAT(op) \
  CUTE_DEVICE float hreduce8_float_ ## op(float x[8]) { return 0.f; }
#endif

DEFINE_HREDUCE8_FLOAT(add)
DEFINE_HREDUCE8_FLOAT(max)
DEFINE_HREDUCE16_FLOAT(add)
DEFINE_HREDUCE16_FLOAT(max)

// Subgroup-cooperative reduction of a SubgroupTensor.
template <int Mode, class BinaryOp,
          class Engine, class FragLayout, class SubgroupTVLayout>
CUTE_HOST_DEVICE
auto
reduce(SubgroupTensor<Engine,FragLayout,SubgroupTVLayout> const& src, BinaryOp op)
{
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  using T = typename Engine::value_type;
  using TVToV = Layout<Shape<intel::_SGSize,int>, Stride<_0,_1>>;

  /* Retrieve logical coordinate -> (T,V) mapping */
  constexpr auto shape = atuple_coshape(SubgroupTVLayout{});
  constexpr auto coord_to_tv = right_inverse(project_strides(SubgroupTVLayout{})).with_shape(shape);

  /* Move reduction coordinate to mode-0 and group the rest in mode-1. Then, remove work-item modes. */
  constexpr auto rcoord_to_tv = make_layout(select<Mode>(coord_to_tv), remove<Mode>(coord_to_tv));
  constexpr auto rcoord_to_v = filter(composition(TVToV{}, rcoord_to_tv), Step<_1,_1>{});

  /* Regroup input tensor */
  Tensor src_r = make_tensor(src.data(), rcoord_to_v);

  /* Create output tensor */
  auto rshape = replace<Mode>(shape, _1{});
  Tensor out = make_subgroup_tensor(make_tensor<T>(ceil_div(size(rshape), intel::_SGSize{})),
                                    make_identity_layout(rshape));

  /* Check for reduction type */
  constexpr bool horizontal = (size<0>(rcoord_to_tv) == intel::_SGSize{} * size<0>(rcoord_to_v));
  constexpr bool vertical   = (size<1>(rcoord_to_tv) == intel::_SGSize{} * size<1>(rcoord_to_v));

  /* Check for optimized reductions */
  constexpr bool align16 = is_constant_v<0, decltype(size<1>(rcoord_to_v) % _16{})>;
  constexpr bool align8  = is_constant_v<8, decltype(size<1>(rcoord_to_v))>;

  constexpr bool hadd = (horizontal && is_same_v<T, float> && is_same_v<BinaryOp, sycl::plus<void>>);
  constexpr bool hmax = (horizontal && is_same_v<T, float> && is_same_v<BinaryOp, sycl::maximum<void>>);

  constexpr bool hadd16 = hadd && align16;
  constexpr bool hmax16 = hmax && align16;

  constexpr bool hadd8 = hadd && align8;
  constexpr bool hmax8 = hmax && align8;

  [[maybe_unused]] T temp[size<1>(rcoord_to_v)];  /* array of partial reductions */

  CUTE_UNROLL
  for (int j = 0; j < size<1>(rcoord_to_v); j++) {
    T acc = src_r(0, j);
    CUTE_UNROLL
    for (int i = 1; i < size<0>(rcoord_to_v); i++) {
      acc = op(acc, src_r(i, j));
    }

    if constexpr (hadd16 || hmax16 || hadd8 || hmax8)
      temp[j] = acc;
    else if constexpr (horizontal)
      set_single_value(out, j, reduce_over_group(sg, acc, op));
    else if constexpr (vertical)
      out(j) = acc;
    else
      static_assert("Unimplemented reduction type");
  }

  if constexpr (hadd16) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(rcoord_to_v); j += 16) {
      out(j/16) = hreduce16_float_add(&temp[j]);
    }
  } else if constexpr (hmax16) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(rcoord_to_v); j += 16) {
      out(j/16) = hreduce16_float_max(&temp[j]);
    }
  } else if constexpr (hadd8) {
    out(0) = hreduce8_float_add(&temp[0]);
  } else if constexpr (hmax8) {
    out(0) = hreduce8_float_max(&temp[0]);
  }

  return out;
}

// Control flag for softmax optimization variant
// 0 = Scalar baseline (separate max, scale, exp2 operations)
// 1 = ASM Options (inline asm exp2, separate loops)
// 2 = Vector Hint option (Forcing IGC to use SIMD16 with ext_vector_type)
// 3 = Fused Softmax ASM (mad + exp2 + hreduce_add in single asm block)
#ifndef CUTLASS_SOFTMAX_VARIANT
#define CUTLASS_SOFTMAX_VARIANT 3
#endif

#ifndef CUTLASS_ENABLE_VECTORIZED_EXP2_ASM
#define CUTLASS_ENABLE_VECTORIZED_EXP2_ASM 1
#endif

// ============================================================================
// ASM Option: Write 16 element vector assembly for Exp2. Apply scalar fall back for remaining elements.
// ============================================================================

#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)

// Apply base-2 exponential to 16 float elements using vISA inline assembly.
// Each element is independently exponentiated: data[i] = exp2(data[i])
CUTE_DEVICE
void inline_asm_exp2_16(float& d0,  float& d1,  float& d2,  float& d3,
                        float& d4,  float& d5,  float& d6,  float& d7,
                        float& d8,  float& d9,  float& d10, float& d11,
                        float& d12, float& d13, float& d14, float& d15) {
  asm volatile (
    "{\n"
    ".decl V0_%=  v_type=G type=F num_elts=16 alias=<%0,0>\n"
    ".decl V1_%=  v_type=G type=F num_elts=16 alias=<%1,0>\n"
    ".decl V2_%=  v_type=G type=F num_elts=16 alias=<%2,0>\n"
    ".decl V3_%=  v_type=G type=F num_elts=16 alias=<%3,0>\n"
    ".decl V4_%=  v_type=G type=F num_elts=16 alias=<%4,0>\n"
    ".decl V5_%=  v_type=G type=F num_elts=16 alias=<%5,0>\n"
    ".decl V6_%=  v_type=G type=F num_elts=16 alias=<%6,0>\n"
    ".decl V7_%=  v_type=G type=F num_elts=16 alias=<%7,0>\n"
    ".decl V8_%=  v_type=G type=F num_elts=16 alias=<%8,0>\n"
    ".decl V9_%=  v_type=G type=F num_elts=16 alias=<%9,0>\n"
    ".decl V10_%= v_type=G type=F num_elts=16 alias=<%10,0>\n"
    ".decl V11_%= v_type=G type=F num_elts=16 alias=<%11,0>\n"
    ".decl V12_%= v_type=G type=F num_elts=16 alias=<%12,0>\n"
    ".decl V13_%= v_type=G type=F num_elts=16 alias=<%13,0>\n"
    ".decl V14_%= v_type=G type=F num_elts=16 alias=<%14,0>\n"
    ".decl V15_%= v_type=G type=F num_elts=16 alias=<%15,0>\n"
    "exp (M1_NM,16) V0_%=(0,0)<1>  V0_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V1_%=(0,0)<1>  V1_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V2_%=(0,0)<1>  V2_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V3_%=(0,0)<1>  V3_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V4_%=(0,0)<1>  V4_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V5_%=(0,0)<1>  V5_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V6_%=(0,0)<1>  V6_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V7_%=(0,0)<1>  V7_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V8_%=(0,0)<1>  V8_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V9_%=(0,0)<1>  V9_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V10_%=(0,0)<1> V10_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V11_%=(0,0)<1> V11_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V12_%=(0,0)<1> V12_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V13_%=(0,0)<1> V13_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V14_%=(0,0)<1> V14_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V15_%=(0,0)<1> V15_%=(0,0)<1;1,0>\n"
    "}\n"
    : "+rw"(d0),  "+rw"(d1),  "+rw"(d2),  "+rw"(d3),
      "+rw"(d4),  "+rw"(d5),  "+rw"(d6),  "+rw"(d7),
      "+rw"(d8),  "+rw"(d9),  "+rw"(d10), "+rw"(d11),
      "+rw"(d12), "+rw"(d13), "+rw"(d14), "+rw"(d15)
  );
}

// Apply base-2 exponential to 8 float elements using vISA inline assembly.
CUTE_DEVICE
void inline_asm_exp2_8(float& d0, float& d1, float& d2, float& d3,
                       float& d4, float& d5, float& d6, float& d7) {
  asm volatile (
    "{\n"
    ".decl V0_%= v_type=G type=F num_elts=16 alias=<%0,0>\n"
    ".decl V1_%= v_type=G type=F num_elts=16 alias=<%1,0>\n"
    ".decl V2_%= v_type=G type=F num_elts=16 alias=<%2,0>\n"
    ".decl V3_%= v_type=G type=F num_elts=16 alias=<%3,0>\n"
    ".decl V4_%= v_type=G type=F num_elts=16 alias=<%4,0>\n"
    ".decl V5_%= v_type=G type=F num_elts=16 alias=<%5,0>\n"
    ".decl V6_%= v_type=G type=F num_elts=16 alias=<%6,0>\n"
    ".decl V7_%= v_type=G type=F num_elts=16 alias=<%7,0>\n"
    "exp (M1_NM,16) V0_%=(0,0)<1> V0_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V1_%=(0,0)<1> V1_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V2_%=(0,0)<1> V2_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V3_%=(0,0)<1> V3_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V4_%=(0,0)<1> V4_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V5_%=(0,0)<1> V5_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V6_%=(0,0)<1> V6_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) V7_%=(0,0)<1> V7_%=(0,0)<1;1,0>\n"
    "}\n"
    : "+rw"(d0), "+rw"(d1), "+rw"(d2), "+rw"(d3),
      "+rw"(d4), "+rw"(d5), "+rw"(d6), "+rw"(d7)
  );
}

// Apply inline asm exp2 to all elements of a fragment, processing in chunks
// of 16 (or 8 for remainder). Falls back to sycl::native::exp2 for odd sizes.
template <typename Fragment>
CUTE_DEVICE
void apply_inline_asm_exp(Fragment& frag) {
  constexpr int N = size(typename Fragment::layout_type{});

  int i = 0;
  // Process full chunks of 16
  CUTE_UNROLL
  for (; i + 16 <= N; i += 16) {
    inline_asm_exp2_16(
      frag(i+0),  frag(i+1),  frag(i+2),  frag(i+3),
      frag(i+4),  frag(i+5),  frag(i+6),  frag(i+7),
      frag(i+8),  frag(i+9),  frag(i+10), frag(i+11),
      frag(i+12), frag(i+13), frag(i+14), frag(i+15));
  }
  // Process remaining chunk of 8
  if constexpr ((N % 16) >= 8) {
    if (i + 8 <= N) {
      inline_asm_exp2_8(
        frag(i+0), frag(i+1), frag(i+2), frag(i+3),
        frag(i+4), frag(i+5), frag(i+6), frag(i+7));
      i += 8;
    }
  }
  // Scalar fallback for any remaining elements
  CUTE_UNROLL
  for (; i < N; i++) {
    frag(i) = sycl::native::exp2(frag(i));
  }
}

#else
// Fallback: apply_inline_asm_exp uses scalar exp2 on non-Intel targets
template <typename Fragment>
CUTE_DEVICE
void apply_inline_asm_exp(Fragment& frag) {
  constexpr int N = size(typename Fragment::layout_type{});
  CUTE_UNROLL
  for (int i = 0; i < N; i++) {
    frag(i) = sycl::native::exp2(frag(i));
  }
}
#endif // __SYCL_DEVICE_ONLY__ && SYCL_INTEL_TARGET

// ============================================================================
// Option 1C: Fused Softmax — mad + exp2 + horizontal sum in single asm block
// ============================================================================
// Fuses scale-subtract, exp2, and row-sum reduction into one inline asm pass.
// Each element: d[i] = exp2(scale * d[i] - max_val), and returns sum(d[0..15]).
// The d[i] values are modified in-place (needed for P*V GEMM).

#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)

// Fused mad + exp2 + hreduce16_add for 16 elements.
// Computes d[i] = exp2(scale * d[i] - max_val) in-place and returns the sum.
CUTE_DEVICE
float fused_mad_exp2_hsum16(
    float& d0,  float& d1,  float& d2,  float& d3,
    float& d4,  float& d5,  float& d6,  float& d7,
    float& d8,  float& d9,  float& d10, float& d11,
    float& d12, float& d13, float& d14, float& d15,
    float scale, float max_val)
{
  float sum;
  asm volatile (
    "{\n"
    // Float aliases for d0-d15 (mad + exp targets)
    ".decl D0_%=  v_type=G type=F num_elts=16 alias=<%0,0>\n"
    ".decl D1_%=  v_type=G type=F num_elts=16 alias=<%1,0>\n"
    ".decl D2_%=  v_type=G type=F num_elts=16 alias=<%2,0>\n"
    ".decl D3_%=  v_type=G type=F num_elts=16 alias=<%3,0>\n"
    ".decl D4_%=  v_type=G type=F num_elts=16 alias=<%4,0>\n"
    ".decl D5_%=  v_type=G type=F num_elts=16 alias=<%5,0>\n"
    ".decl D6_%=  v_type=G type=F num_elts=16 alias=<%6,0>\n"
    ".decl D7_%=  v_type=G type=F num_elts=16 alias=<%7,0>\n"
    ".decl D8_%=  v_type=G type=F num_elts=16 alias=<%8,0>\n"
    ".decl D9_%=  v_type=G type=F num_elts=16 alias=<%9,0>\n"
    ".decl D10_%= v_type=G type=F num_elts=16 alias=<%10,0>\n"
    ".decl D11_%= v_type=G type=F num_elts=16 alias=<%11,0>\n"
    ".decl D12_%= v_type=G type=F num_elts=16 alias=<%12,0>\n"
    ".decl D13_%= v_type=G type=F num_elts=16 alias=<%13,0>\n"
    ".decl D14_%= v_type=G type=F num_elts=16 alias=<%14,0>\n"
    ".decl D15_%= v_type=G type=F num_elts=16 alias=<%15,0>\n"
    // Scale and max_val aliases (input-only)
    ".decl SCALE_%= v_type=G type=F num_elts=16 alias=<%17,0>\n"
    ".decl MAXV_%=  v_type=G type=F num_elts=16 alias=<%18,0>\n"
    // UD aliases for hreduce sel (same registers as D0-D15, reinterpreted as UD)
    ".decl IN0_%=  v_type=G type=UD num_elts=16 alias=<%0,0>\n"
    ".decl IN1_%=  v_type=G type=UD num_elts=16 alias=<%1,0>\n"
    ".decl IN2_%=  v_type=G type=UD num_elts=16 alias=<%2,0>\n"
    ".decl IN3_%=  v_type=G type=UD num_elts=16 alias=<%3,0>\n"
    ".decl IN4_%=  v_type=G type=UD num_elts=16 alias=<%4,0>\n"
    ".decl IN5_%=  v_type=G type=UD num_elts=16 alias=<%5,0>\n"
    ".decl IN6_%=  v_type=G type=UD num_elts=16 alias=<%6,0>\n"
    ".decl IN7_%=  v_type=G type=UD num_elts=16 alias=<%7,0>\n"
    ".decl IN8_%=  v_type=G type=UD num_elts=16 alias=<%8,0>\n"
    ".decl IN9_%=  v_type=G type=UD num_elts=16 alias=<%9,0>\n"
    ".decl IN10_%= v_type=G type=UD num_elts=16 alias=<%10,0>\n"
    ".decl IN11_%= v_type=G type=UD num_elts=16 alias=<%11,0>\n"
    ".decl IN12_%= v_type=G type=UD num_elts=16 alias=<%12,0>\n"
    ".decl IN13_%= v_type=G type=UD num_elts=16 alias=<%13,0>\n"
    ".decl IN14_%= v_type=G type=UD num_elts=16 alias=<%14,0>\n"
    ".decl IN15_%= v_type=G type=UD num_elts=16 alias=<%15,0>\n"
    // Interleave predicate masks
    ".decl INTERLEAVE_2_%= v_type=P num_elts=16\n"
    ".decl INTERLEAVE_4_%= v_type=P num_elts=16\n"
    ".decl INTERLEAVE_8_%= v_type=P num_elts=16\n"
    // Temp registers for hreduce tree reduction
    ".decl RA0_%=  v_type=G type=UD num_elts=32 align=64\n"
    ".decl RA2_%=  v_type=G type=UD num_elts=32 align=64\n"
    ".decl RA4_%=  v_type=G type=UD num_elts=32 align=64\n"
    ".decl RA6_%=  v_type=G type=UD num_elts=32 align=64\n"
    ".decl RA8_%=  v_type=G type=UD num_elts=32 align=64\n"
    ".decl RA10_%= v_type=G type=UD num_elts=32 align=64\n"
    ".decl RA12_%= v_type=G type=UD num_elts=32 align=64\n"
    ".decl RA14_%= v_type=G type=UD num_elts=32 align=64\n"
    ".decl RF0_%=  v_type=G type=F num_elts=16 alias=<RA0_%=,0>\n"
    ".decl RF1_%=  v_type=G type=F num_elts=16 alias=<RA0_%=,64>\n"
    ".decl RF2_%=  v_type=G type=F num_elts=16 alias=<RA2_%=,0>\n"
    ".decl RF3_%=  v_type=G type=F num_elts=16 alias=<RA2_%=,64>\n"
    ".decl RF4_%=  v_type=G type=F num_elts=16 alias=<RA4_%=,0>\n"
    ".decl RF5_%=  v_type=G type=F num_elts=16 alias=<RA4_%=,64>\n"
    ".decl RF6_%=  v_type=G type=F num_elts=16 alias=<RA6_%=,0>\n"
    ".decl RF7_%=  v_type=G type=F num_elts=16 alias=<RA6_%=,64>\n"
    ".decl RF8_%=  v_type=G type=F num_elts=16 alias=<RA8_%=,0>\n"
    ".decl RF9_%=  v_type=G type=F num_elts=16 alias=<RA8_%=,64>\n"
    ".decl RF10_%= v_type=G type=F num_elts=16 alias=<RA10_%=,0>\n"
    ".decl RF11_%= v_type=G type=F num_elts=16 alias=<RA10_%=,64>\n"
    ".decl RF12_%= v_type=G type=F num_elts=16 alias=<RA12_%=,0>\n"
    ".decl RF13_%= v_type=G type=F num_elts=16 alias=<RA12_%=,64>\n"
    ".decl RF14_%= v_type=G type=F num_elts=16 alias=<RA14_%=,0>\n"
    ".decl RF15_%= v_type=G type=F num_elts=16 alias=<RA14_%=,64>\n"
    //
    // === Phase 1: MAD — d[i] = scale * d[i] - max_val ===
    //
    "mad (M1_NM,16) D0_%=(0,0)<1>  SCALE_%=(0,0)<0;1,0> D0_%=(0,0)<1;1,0>  (-)MAXV_%=(0,0)<0;1,0>\n"
    "mad (M1_NM,16) D1_%=(0,0)<1>  SCALE_%=(0,0)<0;1,0> D1_%=(0,0)<1;1,0>  (-)MAXV_%=(0,0)<0;1,0>\n"
    "mad (M1_NM,16) D2_%=(0,0)<1>  SCALE_%=(0,0)<0;1,0> D2_%=(0,0)<1;1,0>  (-)MAXV_%=(0,0)<0;1,0>\n"
    "mad (M1_NM,16) D3_%=(0,0)<1>  SCALE_%=(0,0)<0;1,0> D3_%=(0,0)<1;1,0>  (-)MAXV_%=(0,0)<0;1,0>\n"
    "mad (M1_NM,16) D4_%=(0,0)<1>  SCALE_%=(0,0)<0;1,0> D4_%=(0,0)<1;1,0>  (-)MAXV_%=(0,0)<0;1,0>\n"
    "mad (M1_NM,16) D5_%=(0,0)<1>  SCALE_%=(0,0)<0;1,0> D5_%=(0,0)<1;1,0>  (-)MAXV_%=(0,0)<0;1,0>\n"
    "mad (M1_NM,16) D6_%=(0,0)<1>  SCALE_%=(0,0)<0;1,0> D6_%=(0,0)<1;1,0>  (-)MAXV_%=(0,0)<0;1,0>\n"
    "mad (M1_NM,16) D7_%=(0,0)<1>  SCALE_%=(0,0)<0;1,0> D7_%=(0,0)<1;1,0>  (-)MAXV_%=(0,0)<0;1,0>\n"
    "mad (M1_NM,16) D8_%=(0,0)<1>  SCALE_%=(0,0)<0;1,0> D8_%=(0,0)<1;1,0>  (-)MAXV_%=(0,0)<0;1,0>\n"
    "mad (M1_NM,16) D9_%=(0,0)<1>  SCALE_%=(0,0)<0;1,0> D9_%=(0,0)<1;1,0>  (-)MAXV_%=(0,0)<0;1,0>\n"
    "mad (M1_NM,16) D10_%=(0,0)<1> SCALE_%=(0,0)<0;1,0> D10_%=(0,0)<1;1,0> (-)MAXV_%=(0,0)<0;1,0>\n"
    "mad (M1_NM,16) D11_%=(0,0)<1> SCALE_%=(0,0)<0;1,0> D11_%=(0,0)<1;1,0> (-)MAXV_%=(0,0)<0;1,0>\n"
    "mad (M1_NM,16) D12_%=(0,0)<1> SCALE_%=(0,0)<0;1,0> D12_%=(0,0)<1;1,0> (-)MAXV_%=(0,0)<0;1,0>\n"
    "mad (M1_NM,16) D13_%=(0,0)<1> SCALE_%=(0,0)<0;1,0> D13_%=(0,0)<1;1,0> (-)MAXV_%=(0,0)<0;1,0>\n"
    "mad (M1_NM,16) D14_%=(0,0)<1> SCALE_%=(0,0)<0;1,0> D14_%=(0,0)<1;1,0> (-)MAXV_%=(0,0)<0;1,0>\n"
    "mad (M1_NM,16) D15_%=(0,0)<1> SCALE_%=(0,0)<0;1,0> D15_%=(0,0)<1;1,0> (-)MAXV_%=(0,0)<0;1,0>\n"
    //
    // === Phase 2: EXP — d[i] = exp2(d[i]) ===
    //
    "exp (M1_NM,16) D0_%=(0,0)<1>  D0_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) D1_%=(0,0)<1>  D1_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) D2_%=(0,0)<1>  D2_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) D3_%=(0,0)<1>  D3_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) D4_%=(0,0)<1>  D4_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) D5_%=(0,0)<1>  D5_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) D6_%=(0,0)<1>  D6_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) D7_%=(0,0)<1>  D7_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) D8_%=(0,0)<1>  D8_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) D9_%=(0,0)<1>  D9_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) D10_%=(0,0)<1> D10_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) D11_%=(0,0)<1> D11_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) D12_%=(0,0)<1> D12_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) D13_%=(0,0)<1> D13_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) D14_%=(0,0)<1> D14_%=(0,0)<1;1,0>\n"
    "exp (M1_NM,16) D15_%=(0,0)<1> D15_%=(0,0)<1;1,0>\n"
    //
    // === Phase 3: HREDUCE ADD — sum the 16 exp2'd values ===
    // IN aliases read exp2'd values from D registers (non-destructive).
    //
    "setp (M1_NM,16) INTERLEAVE_2_%= 0x5555:uw\n"
    "setp (M1_NM,16) INTERLEAVE_4_%= 0x3333:uw\n"
    "setp (M1_NM,16) INTERLEAVE_8_%= 0x0F0F:uw\n"
    // Round 1: interleave pairs
    "(!INTERLEAVE_2_%=) sel (M1_NM,16) RA0_%=(0,0)<1>   IN1_%=(0,0)<2;2,0>   IN0_%=(0,0)<1;1,0>\n"
    " (INTERLEAVE_2_%=) sel (M1_NM,16) RA0_%=(1,0)<1>   IN0_%=(0,1)<2;2,0>   IN1_%=(0,0)<1;1,0>\n"
    "(!INTERLEAVE_2_%=) sel (M1_NM,16) RA2_%=(0,0)<1>   IN3_%=(0,0)<2;2,0>   IN2_%=(0,0)<1;1,0>\n"
    " (INTERLEAVE_2_%=) sel (M1_NM,16) RA2_%=(1,0)<1>   IN2_%=(0,1)<2;2,0>   IN3_%=(0,0)<1;1,0>\n"
    "(!INTERLEAVE_2_%=) sel (M1_NM,16) RA4_%=(0,0)<1>   IN5_%=(0,0)<2;2,0>   IN4_%=(0,0)<1;1,0>\n"
    " (INTERLEAVE_2_%=) sel (M1_NM,16) RA4_%=(1,0)<1>   IN4_%=(0,1)<2;2,0>   IN5_%=(0,0)<1;1,0>\n"
    "(!INTERLEAVE_2_%=) sel (M1_NM,16) RA6_%=(0,0)<1>   IN7_%=(0,0)<2;2,0>   IN6_%=(0,0)<1;1,0>\n"
    " (INTERLEAVE_2_%=) sel (M1_NM,16) RA6_%=(1,0)<1>   IN6_%=(0,1)<2;2,0>   IN7_%=(0,0)<1;1,0>\n"
    "(!INTERLEAVE_2_%=) sel (M1_NM,16) RA8_%=(0,0)<1>   IN9_%=(0,0)<2;2,0>   IN8_%=(0,0)<1;1,0>\n"
    " (INTERLEAVE_2_%=) sel (M1_NM,16) RA8_%=(1,0)<1>   IN8_%=(0,1)<2;2,0>   IN9_%=(0,0)<1;1,0>\n"
    "(!INTERLEAVE_2_%=) sel (M1_NM,16) RA10_%=(0,0)<1>  IN11_%=(0,0)<2;2,0>  IN10_%=(0,0)<1;1,0>\n"
    " (INTERLEAVE_2_%=) sel (M1_NM,16) RA10_%=(1,0)<1>  IN10_%=(0,1)<2;2,0>  IN11_%=(0,0)<1;1,0>\n"
    "(!INTERLEAVE_2_%=) sel (M1_NM,16) RA12_%=(0,0)<1>  IN13_%=(0,0)<2;2,0>  IN12_%=(0,0)<1;1,0>\n"
    " (INTERLEAVE_2_%=) sel (M1_NM,16) RA12_%=(1,0)<1>  IN12_%=(0,1)<2;2,0>  IN13_%=(0,0)<1;1,0>\n"
    "(!INTERLEAVE_2_%=) sel (M1_NM,16) RA14_%=(0,0)<1>  IN15_%=(0,0)<2;2,0>  IN14_%=(0,0)<1;1,0>\n"
    " (INTERLEAVE_2_%=) sel (M1_NM,16) RA14_%=(1,0)<1>  IN14_%=(0,1)<2;2,0>  IN15_%=(0,0)<1;1,0>\n"
    // Reduce round 1
    "add (M1_NM,16) RF0_%=(0,0)<1>  RF0_%=(0,0)<1;1,0>  RF1_%=(0,0)<1;1,0>\n"
    "add (M1_NM,16) RF3_%=(0,0)<1>  RF2_%=(0,0)<1;1,0>  RF3_%=(0,0)<1;1,0>\n"
    "add (M1_NM,16) RF4_%=(0,0)<1>  RF4_%=(0,0)<1;1,0>  RF5_%=(0,0)<1;1,0>\n"
    "add (M1_NM,16) RF7_%=(0,0)<1>  RF6_%=(0,0)<1;1,0>  RF7_%=(0,0)<1;1,0>\n"
    "add (M1_NM,16) RF8_%=(0,0)<1>  RF8_%=(0,0)<1;1,0>  RF9_%=(0,0)<1;1,0>\n"
    "add (M1_NM,16) RF11_%=(0,0)<1> RF10_%=(0,0)<1;1,0> RF11_%=(0,0)<1;1,0>\n"
    "add (M1_NM,16) RF12_%=(0,0)<1> RF12_%=(0,0)<1;1,0> RF13_%=(0,0)<1;1,0>\n"
    "add (M1_NM,16) RF15_%=(0,0)<1> RF14_%=(0,0)<1;1,0> RF15_%=(0,0)<1;1,0>\n"
    // Round 2: interleave quads
    "(!INTERLEAVE_4_%=) sel (M1_NM,16) RA0_%=(1,0)<1>   RA2_%=(0,14)<1;1,0>  RA0_%=(0,0)<1;1,0>\n"
    " (INTERLEAVE_4_%=) sel (M1_NM,16) RA0_%=(0,0)<1>   RA0_%=(0,2)<1;1,0>   RA2_%=(1,0)<1;1,0>\n"
    "(!INTERLEAVE_4_%=) sel (M1_NM,16) RA4_%=(1,0)<1>   RA6_%=(0,14)<1;1,0>  RA4_%=(0,0)<1;1,0>\n"
    " (INTERLEAVE_4_%=) sel (M1_NM,16) RA4_%=(0,0)<1>   RA4_%=(0,2)<1;1,0>   RA6_%=(1,0)<1;1,0>\n"
    "(!INTERLEAVE_4_%=) sel (M1_NM,16) RA8_%=(1,0)<1>   RA10_%=(0,14)<1;1,0> RA8_%=(0,0)<1;1,0>\n"
    " (INTERLEAVE_4_%=) sel (M1_NM,16) RA8_%=(0,0)<1>   RA8_%=(0,2)<1;1,0>   RA10_%=(1,0)<1;1,0>\n"
    "(!INTERLEAVE_4_%=) sel (M1_NM,16) RA12_%=(1,0)<1>  RA14_%=(0,14)<1;1,0> RA12_%=(0,0)<1;1,0>\n"
    " (INTERLEAVE_4_%=) sel (M1_NM,16) RA12_%=(0,0)<1>  RA12_%=(0,2)<1;1,0>  RA14_%=(1,0)<1;1,0>\n"
    // Reduce round 2
    "add (M1_NM,16) RF0_%=(0,0)<1>  RF0_%=(0,0)<1;1,0>  RF1_%=(0,0)<1;1,0>\n"
    "add (M1_NM,16) RF5_%=(0,0)<1>  RF4_%=(0,0)<1;1,0>  RF5_%=(0,0)<1;1,0>\n"
    "add (M1_NM,16) RF8_%=(0,0)<1>  RF8_%=(0,0)<1;1,0>  RF9_%=(0,0)<1;1,0>\n"
    "add (M1_NM,16) RF13_%=(0,0)<1> RF12_%=(0,0)<1;1,0> RF13_%=(0,0)<1;1,0>\n"
    // Round 3: interleave octets
    "(!INTERLEAVE_8_%=) sel (M1_NM,16) RA0_%=(1,0)<1>  RA4_%=(0,12)<1;1,0>  RA0_%=(0,0)<1;1,0>\n"
    " (INTERLEAVE_8_%=) sel (M1_NM,16) RA0_%=(0,0)<1>  RA0_%=(0,4)<1;1,0>   RA4_%=(1,0)<1;1,0>\n"
    "(!INTERLEAVE_8_%=) sel (M1_NM,16) RA8_%=(1,0)<1>  RA12_%=(0,12)<1;1,0> RA8_%=(0,0)<1;1,0>\n"
    " (INTERLEAVE_8_%=) sel (M1_NM,16) RA8_%=(0,0)<1>  RA8_%=(0,4)<1;1,0>   RA12_%=(1,0)<1;1,0>\n"
    // Reduce round 3
    "add (M1_NM,16) RF0_%=(0,0)<1>  RF0_%=(0,0)<1;1,0>  RF1_%=(0,0)<1;1,0>\n"
    "add (M1_NM,16) RF8_%=(0,0)<1>  RF8_%=(0,0)<1;1,0>  RF9_%=(0,0)<1;1,0>\n"
    // Round 4: final half reduction
    "mov (M1_NM, 8) RA0_%=(1,0)<1>  RA0_%=(0,8)<1;1,0>\n"
    "mov (M1_NM, 8) RA8_%=(1,8)<1>  RA8_%=(0,0)<1;1,0>\n"
    "add (M1_NM,8) %16(0,0)<1> RF0_%=(0,0)<1;1,0> RF1_%=(0,0)<1;1,0>\n"
    "add (M1_NM,8) %16(0,8)<1> RF8_%=(0,8)<1;1,0> RF9_%=(0,8)<1;1,0>\n"
    "}\n"
    : "+rw"(d0),  "+rw"(d1),  "+rw"(d2),  "+rw"(d3),
      "+rw"(d4),  "+rw"(d5),  "+rw"(d6),  "+rw"(d7),
      "+rw"(d8),  "+rw"(d9),  "+rw"(d10), "+rw"(d11),
      "+rw"(d12), "+rw"(d13), "+rw"(d14), "+rw"(d15),
      "=rw"(sum)
    : "rw"(scale), "rw"(max_val)
  );
  return sum;
}

// Fused softmax: applies scale-subtract-exp2 in-place and reduces by sum.
// Mirrors reduce<1> layout logic but fuses scale*x-max, exp2, and hreduce add.
// Requirements: Mode=1, horizontal reduce, float type, align16.
// Falls back to separate ops if requirements not met.
template <int Mode, class Engine, class FragLayout, class SubgroupTVLayout,
          class FragRow>
CUTE_DEVICE
auto
fused_softmax_scale_exp_sum(
    SubgroupTensor<Engine,FragLayout,SubgroupTVLayout>& src,
    float scale,
    FragRow const& row_max)
{
  using T = typename Engine::value_type;
  using TVToV = Layout<Shape<intel::_SGSize,int>, Stride<_0,_1>>;

  constexpr auto shape = atuple_coshape(SubgroupTVLayout{});
  constexpr auto coord_to_tv = right_inverse(project_strides(SubgroupTVLayout{})).with_shape(shape);

  constexpr auto rcoord_to_tv = make_layout(select<Mode>(coord_to_tv), remove<Mode>(coord_to_tv));
  constexpr auto rcoord_to_v = filter(composition(TVToV{}, rcoord_to_tv), Step<_1,_1>{});

  Tensor src_r = make_tensor(src.data(), rcoord_to_v);

  auto rshape = replace<Mode>(shape, _1{});
  Tensor out = make_subgroup_tensor(make_tensor<T>(ceil_div(size(rshape), intel::_SGSize{})),
                                    make_identity_layout(rshape));

  constexpr bool horizontal = (size<0>(rcoord_to_tv) == intel::_SGSize{} * size<0>(rcoord_to_v));
  constexpr bool align16 = is_constant_v<0, decltype(size<1>(rcoord_to_v) % _16{})>;
  constexpr bool can_fuse = (horizontal && is_same_v<T, float> && align16 &&
                             size<0>(rcoord_to_v) == 1);

  if constexpr (can_fuse) {
    // Optimal path: fused mad+exp2+hsum in single asm block per 16-element row
    CUTE_UNROLL
    for (int j = 0; j < size<1>(rcoord_to_v); j += 16) {
      T row_max_val = row_max(j / 16);
      out(j / 16) = fused_mad_exp2_hsum16(
        src_r(0, j+0),  src_r(0, j+1),  src_r(0, j+2),  src_r(0, j+3),
        src_r(0, j+4),  src_r(0, j+5),  src_r(0, j+6),  src_r(0, j+7),
        src_r(0, j+8),  src_r(0, j+9),  src_r(0, j+10), src_r(0, j+11),
        src_r(0, j+12), src_r(0, j+13), src_r(0, j+14), src_r(0, j+15),
        scale, row_max_val);
    }
  } else {
    // Fallback: separate scale-subtract, exp2, and reduce
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    T temp[size<1>(rcoord_to_v)];

    CUTE_UNROLL
    for (int j = 0; j < size<1>(rcoord_to_v); j++) {
      // Apply scale-subtract and exp2 per element
      CUTE_UNROLL
      for (int i = 0; i < size<0>(rcoord_to_v); i++) {
        // Determine row index for this element
        // For the fallback path, use scalar operations
        src_r(i, j) = sycl::native::exp2(scale * src_r(i, j) - row_max(j / 16));
      }
      T acc = src_r(0, j);
      CUTE_UNROLL
      for (int i = 1; i < size<0>(rcoord_to_v); i++) {
        acc = acc + src_r(i, j);
      }

      if constexpr (horizontal && is_same_v<T, float> && align16)
        temp[j] = acc;
      else if constexpr (horizontal)
        set_single_value(out, j, reduce_over_group(sg, acc, sycl::plus<void>{}));
      else
        out(j) = acc;
    }

    if constexpr (horizontal && is_same_v<T, float> && align16) {
      CUTE_UNROLL
      for (int j = 0; j < size<1>(rcoord_to_v); j += 16) {
        out(j/16) = hreduce16_float_add(&temp[j]);
      }
    }
  }

  return out;
}

#else
// Fallback for non-Intel targets: scalar fused softmax
template <int Mode, class Engine, class FragLayout, class SubgroupTVLayout,
          class FragRow>
CUTE_DEVICE
auto
fused_softmax_scale_exp_sum(
    SubgroupTensor<Engine,FragLayout,SubgroupTVLayout>& src,
    float scale,
    FragRow const& row_max)
{
  using T = typename Engine::value_type;
  // Apply scale-subtract-exp2 in-place
  constexpr int N = size(typename SubgroupTensor<Engine,FragLayout,SubgroupTVLayout>::layout_type{});
  CUTE_UNROLL
  for (int i = 0; i < N; i++) {
    src(i) = sycl::native::exp2(scale * src(i) - row_max(i / 16));  // approximate
  }
  return reduce<Mode>(src, sycl::plus<void>{});
}
#endif // __SYCL_DEVICE_ONLY__ && SYCL_INTEL_TARGET

// ============================================================================
// Option 1B: Vectorized Exp2 Operations
// ============================================================================
// Vectorized exp2 operations using Intel GPU ext_vector_type
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET) && CUTLASS_ENABLE_VECTORIZED_EXP2_ASM

// Use Intel's ext_vector_type for SIMD operations
// This approach avoids complex inline assembly and lets the compiler generate optimal code
template <int N>
struct VectorExp2Helper {
  using VecType = float __attribute__((ext_vector_type(N)));
  
  CUTE_DEVICE
  static void apply(float* data) {
    // Load data into vector register (compiler optimizes this)
    VecType vec;
    CUTE_UNROLL
    for (int i = 0; i < N; i++) {
      vec[i] = data[i];
    }
    
    // Apply exp2 element-wise (compiler should vectorize this)
    CUTE_UNROLL
    for (int i = 0; i < N; i++) {
      vec[i] = sycl::native::exp2(vec[i]);
    }
    
    // Store back (compiler optimizes this)
    CUTE_UNROLL
    for (int i = 0; i < N; i++) {
      data[i] = vec[i];
    }
  }
};

// Vectorized exp2 for 16 float elements
CUTE_DEVICE
void vectorized_exp2_16(float* data) {
  VectorExp2Helper<16>::apply(data);
  
  // OLD INLINE ASSEMBLY VERSION (commented out - had accuracy issues)
  // Keeping for reference in case we want to investigate further
  /*
  asm volatile (
    "{\n"
    ".decl IN0 v_type=G type=F num_elts=16 alias=<%0,0>\n"
    ".decl IN1 v_type=G type=F num_elts=16 alias=<%1,0>\n"
    ".decl IN2 v_type=G type=F num_elts=16 alias=<%2,0>\n"
    ".decl IN3 v_type=G type=F num_elts=16 alias=<%3,0>\n"
    ".decl IN4 v_type=G type=F num_elts=16 alias=<%4,0>\n"
    ".decl IN5 v_type=G type=F num_elts=16 alias=<%5,0>\n"
    ".decl IN6 v_type=G type=F num_elts=16 alias=<%6,0>\n"
    ".decl IN7 v_type=G type=F num_elts=16 alias=<%7,0>\n"
    ".decl IN8 v_type=G type=F num_elts=16 alias=<%8,0>\n"
    ".decl IN9 v_type=G type=F num_elts=16 alias=<%9,0>\n"
    ".decl IN10 v_type=G type=F num_elts=16 alias=<%10,0>\n"
    ".decl IN11 v_type=G type=F num_elts=16 alias=<%11,0>\n"
    ".decl IN12 v_type=G type=F num_elts=16 alias=<%12,0>\n"
    ".decl IN13 v_type=G type=F num_elts=16 alias=<%13,0>\n"
    ".decl IN14 v_type=G type=F num_elts=16 alias=<%14,0>\n"
    ".decl IN15 v_type=G type=F num_elts=16 alias=<%15,0>\n"
    ".decl VEXP v_type=G type=F num_elts=16\n"
    // Gather inputs into vector register
    "mov (M1,1) VEXP(0,0)<1> IN0(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,1)<1> IN1(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,2)<1> IN2(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,3)<1> IN3(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,4)<1> IN4(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,5)<1> IN5(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,6)<1> IN6(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,7)<1> IN7(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,8)<1> IN8(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,9)<1> IN9(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,10)<1> IN10(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,11)<1> IN11(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,12)<1> IN12(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,13)<1> IN13(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,14)<1> IN14(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,15)<1> IN15(0,0)<0;1,0>\n"
    // Apply exp2
    "exp (M1_NM,16) VEXP(0,0)<1> VEXP(0,0)<1;1,0>\n"
    // Scatter results back
    "mov (M1,1) IN0(0,0)<1> VEXP(0,0)<0;1,0>\n"
    "mov (M1,1) IN1(0,0)<1> VEXP(0,1)<0;1,0>\n"
    "mov (M1,1) IN2(0,0)<1> VEXP(0,2)<0;1,0>\n"
    "mov (M1,1) IN3(0,0)<1> VEXP(0,3)<0;1,0>\n"
    "mov (M1,1) IN4(0,0)<1> VEXP(0,4)<0;1,0>\n"
    "mov (M1,1) IN5(0,0)<1> VEXP(0,5)<0;1,0>\n"
    "mov (M1,1) IN6(0,0)<1> VEXP(0,6)<0;1,0>\n"
    "mov (M1,1) IN7(0,0)<1> VEXP(0,7)<0;1,0>\n"
    "mov (M1,1) IN8(0,0)<1> VEXP(0,8)<0;1,0>\n"
    "mov (M1,1) IN9(0,0)<1> VEXP(0,9)<0;1,0>\n"
    "mov (M1,1) IN10(0,0)<1> VEXP(0,10)<0;1,0>\n"
    "mov (M1,1) IN11(0,0)<1> VEXP(0,11)<0;1,0>\n"
    "mov (M1,1) IN12(0,0)<1> VEXP(0,12)<0;1,0>\n"
    "mov (M1,1) IN13(0,0)<1> VEXP(0,13)<0;1,0>\n"
    "mov (M1,1) IN14(0,0)<1> VEXP(0,14)<0;1,0>\n"
    "mov (M1,1) IN15(0,0)<1> VEXP(0,15)<0;1,0>\n"
    "}\n"
    : "+rw"(data[0]), "+rw"(data[1]), "+rw"(data[2]),  "+rw"(data[3]),
      "+rw"(data[4]), "+rw"(data[5]), "+rw"(data[6]),  "+rw"(data[7]),
      "+rw"(data[8]), "+rw"(data[9]), "+rw"(data[10]), "+rw"(data[11]),
      "+rw"(data[12]), "+rw"(data[13]), "+rw"(data[14]), "+rw"(data[15])
  );
  */
}

// Vectorized exp2 for 8 float elements
CUTE_DEVICE
void vectorized_exp2_8(float* data) {
  VectorExp2Helper<8>::apply(data);
  
  // OLD INLINE ASSEMBLY VERSION (commented out - had accuracy issues)
  /*
  asm volatile (
    "{\n"
    ".decl IN0 v_type=G type=F num_elts=16 alias=<%0,0>\n"
    ".decl IN1 v_type=G type=F num_elts=16 alias=<%1,0>\n"
    ".decl IN2 v_type=G type=F num_elts=16 alias=<%2,0>\n"
    ".decl IN3 v_type=G type=F num_elts=16 alias=<%3,0>\n"
    ".decl IN4 v_type=G type=F num_elts=16 alias=<%4,0>\n"
    ".decl IN5 v_type=G type=F num_elts=16 alias=<%5,0>\n"
    ".decl IN6 v_type=G type=F num_elts=16 alias=<%6,0>\n"
    ".decl IN7 v_type=G type=F num_elts=16 alias=<%7,0>\n"
    ".decl VEXP v_type=G type=F num_elts=16\n"
    // Gather inputs into vector register
    "mov (M1,1) VEXP(0,0)<1> IN0(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,1)<1> IN1(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,2)<1> IN2(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,3)<1> IN3(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,4)<1> IN4(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,5)<1> IN5(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,6)<1> IN6(0,0)<0;1,0>\n"
    "mov (M1,1) VEXP(0,7)<1> IN7(0,0)<0;1,0>\n"
    // Apply exp2 to first 8 elements
    "exp (M1_NM,8) VEXP(0,0)<1> VEXP(0,0)<1;1,0>\n"
    // Scatter results back
    "mov (M1,1) IN0(0,0)<1> VEXP(0,0)<0;1,0>\n"
    "mov (M1,1) IN1(0,0)<1> VEXP(0,1)<0;1,0>\n"
    "mov (M1,1) IN2(0,0)<1> VEXP(0,2)<0;1,0>\n"
    "mov (M1,1) IN3(0,0)<1> VEXP(0,3)<0;1,0>\n"
    "mov (M1,1) IN4(0,0)<1> VEXP(0,4)<0;1,0>\n"
    "mov (M1,1) IN5(0,0)<1> VEXP(0,5)<0;1,0>\n"
    "mov (M1,1) IN6(0,0)<1> VEXP(0,6)<0;1,0>\n"
    "mov (M1,1) IN7(0,0)<1> VEXP(0,7)<0;1,0>\n"
    "}\n"
    : "+rw"(data[0]), "+rw"(data[1]), "+rw"(data[2]), "+rw"(data[3]),
      "+rw"(data[4]), "+rw"(data[5]), "+rw"(data[6]), "+rw"(data[7])
  );
  */
}

// Vectorized exp2 for arbitrary size (dispatches to optimized versions)
template <int N>
CUTE_DEVICE
void vectorized_exp2_n(float* data) {
  if constexpr (N == 16) {
    vectorized_exp2_16(data);
  } else if constexpr (N == 8) {
    vectorized_exp2_8(data);
  } else if constexpr (N % 16 == 0) {
    CUTE_UNROLL
    for (int i = 0; i < N; i += 16) {
      vectorized_exp2_16(&data[i]);
    }
  } else if constexpr (N % 8 == 0) {
    CUTE_UNROLL
    for (int i = 0; i < N; i += 8) {
      vectorized_exp2_8(&data[i]);
    }
  } else {
    // Fallback for non-vectorizable sizes
    CUTE_UNROLL
    for (int i = 0; i < N; i++) {
      data[i] = sycl::native::exp2(data[i]);
    }
  }
}

#else
// Fallback implementations (host, non-Intel, or assembly disabled)
// Uses scalar sycl::native::exp2 which is accurate and lets the compiler optimize
CUTE_DEVICE
void vectorized_exp2_16(float* data) {
  for (int i = 0; i < 16; i++) {
    data[i] = sycl::native::exp2(data[i]);
  }
}

CUTE_DEVICE
void vectorized_exp2_8(float* data) {
  for (int i = 0; i < 8; i++) {
    data[i] = sycl::native::exp2(data[i]);
  }
}

template <int N>
CUTE_DEVICE
void vectorized_exp2_n(float* data) {
  for (int i = 0; i < N; i++) {
    data[i] = sycl::native::exp2(data[i]);
  }
}
#endif

// Apply vectorized exp2 to a fragment
// NOTE: Currently uses scalar fallback (CUTLASS_ENABLE_VECTORIZED_EXP2_ASM=0)
//       due to accuracy issues with inline assembly implementation.
//       Set CUTLASS_ENABLE_VECTORIZED_EXP2_ASM=1 to enable assembly version (experimental).
template <typename Fragment>
CUTE_DEVICE
void apply_vectorized_exp2(Fragment& frag) {
  using T = typename Fragment::value_type;
  // Get size at compile time using decltype and cute's size function 
  auto constexpr frag_size = size(typename Fragment::layout_type{});
  constexpr int N = int(frag_size);
  
  if constexpr (is_same_v<T, float>) {
    // For float fragments, we can vectorize if size is right
    if constexpr (N == 16) {
      float temp[16];
      CUTE_UNROLL
      for (int i = 0; i < 16; i++) temp[i] = frag(i);
      vectorized_exp2_16(temp);
      CUTE_UNROLL
      for (int i = 0; i < 16; i++) frag(i) = temp[i];
    } else if constexpr (N == 8) {
      float temp[8];
      CUTE_UNROLL
      for (int i = 0; i < 8; i++) temp[i] = frag(i);
      vectorized_exp2_8(temp);
      CUTE_UNROLL
      for (int i = 0; i < 8; i++) frag(i) = temp[i];
    } else if constexpr (N % 16 == 0) {
      float temp[16];
      CUTE_UNROLL
      for (int chunk = 0; chunk < N / 16; chunk++) {
        CUTE_UNROLL
        for (int i = 0; i < 16; i++) temp[i] = frag(chunk * 16 + i);
        vectorized_exp2_16(temp);
        CUTE_UNROLL
        for (int i = 0; i < 16; i++) frag(chunk * 16 + i) = temp[i];
      }
    } else if constexpr (N % 8 == 0) {
      float temp[8];
      CUTE_UNROLL
      for (int chunk = 0; chunk < N / 8; chunk++) {
        CUTE_UNROLL
        for (int i = 0; i < 8; i++) temp[i] = frag(chunk * 8 + i);
        vectorized_exp2_8(temp);
        CUTE_UNROLL
        for (int i = 0; i < 8; i++) frag(chunk * 8 + i) = temp[i];
      }
    } else {
      // Fallback to scalar operations
      CUTE_UNROLL
      for (int i = 0; i < N; i++) {
        frag(i) = sycl::native::exp2(frag(i));
      }
    }
  } else {
    // Non-float types: fallback to scalar
    CUTE_UNROLL
    for (int i = 0; i < N; i++) {
      frag(i) = sycl::native::exp2(frag(i));
    }
  }
}

} // namespace cute
