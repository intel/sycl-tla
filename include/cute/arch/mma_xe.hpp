/***************************************************************************************************
 * Copyright (c) 2025 ----
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

#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
#define CUTE_ARCH_MMA_XE_ENABLED
#endif

#include <cute/config.hpp>
#include <cute/arch/mma.hpp>
#include <cute/util/sycl_vec.hpp>

namespace cute {

template <int M, typename TypeD, typename TypeA, typename TypeB = TypeA, typename TypeC = TypeD>
struct XE_DPAS_TT;

template <int M, typename TypeD, typename TypeA, typename TypeB, typename TypeC>
struct XE_DPAS_TT_Base
{
  static constexpr int K = 256 / cute::max(sizeof_bits_v<TypeA>, sizeof_bits_v<TypeB>);

  using DVector = intel::vector_t<TypeD, M>;
  using AVector = intel::vector_t<TypeA, (M * K + 15) / 16>;
  using BVector = intel::vector_t<TypeB, K>;
  using CVector = intel::vector_t<TypeC, M>;

  using DRegisters = DVector[1];
  using ARegisters = AVector[1];
  using BRegisters = BVector[1];
  using CRegisters = CVector[1];
};

namespace detail {

using DPASType_f = float;
using DPASType_tf32 = tfloat32_t;
using DPASType_bf = bfloat16_t;
using DPASType_hf = half_t;
using DPASType_ud = uint32_t;
using DPASType_d = int32_t;
using DPASType_u8 = uint8_t;
using DPASType_s8 = int8_t;
using DPASType_u4 = uint4_t;
using DPASType_s4 = int4_t;

}; /* namespace detail */

#ifdef CUTE_ARCH_MMA_XE_ENABLED

#define CUTE_DECLARE_XE_DPAS_TT(M, M8, M16, TD, TA, TB, TC) \
template <> struct XE_DPAS_TT<M, detail::DPASType_ ## TD, detail::DPASType_ ## TA, detail::DPASType_ ## TB, detail::DPASType_ ## TC> \
  : public XE_DPAS_TT_Base<M, detail::DPASType_ ## TD, detail::DPASType_ ## TA, detail::DPASType_ ## TB, detail::DPASType_ ## TC> { \
  CUTE_HOST_DEVICE static void \
  fma(DVector& d, AVector const& a, BVector const& b, CVector const& c) { \
    asm( \
      "{\n" \
      ".decl DST     v_type=G type=" #TD " num_elts=" #M16 " alias=<%0,0>\n" \
      ".decl SRC0    v_type=G type=" #TC " num_elts=" #M16 " alias=<%3,0>\n" \
      ".decl SRC1_UD v_type=G type=UD num_elts=128 alias=<%2,0>\n" \
      ".decl SRC2_UD v_type=G type=UD num_elts=" #M8 " alias=<%1,0>\n" \
      "dpas." #TB "." #TA ".8." #M " (M1, 16) DST.0 SRC0.0 SRC1_UD.0 SRC2_UD(0,0)\n" \
      "}\n" \
      : "=rw"(d) : "rw"(a), "rw"(b), "rw"(c) \
    ); \
  } \
};

#else /* !defined(CUTE_ARCH_MMA_XE_ENABLED) */

#define CUTE_DECLARE_XE_DPAS_TT(M, M8, M16, TD, TA, TB, TC) \
template <> struct XE_DPAS_TT<M, detail::DPASType_ ## TD, detail::DPASType_ ## TA, detail::DPASType_ ## TB, detail::DPASType_ ## TC> \
  : public XE_DPAS_TT_Base<M, detail::DPASType_ ## TD, detail::DPASType_ ## TA, detail::DPASType_ ## TB, detail::DPASType_ ## TC> { \
  CUTE_HOST_DEVICE static void \
  fma(DVector& d, AVector const& a, BVector const& b, CVector const& c) { \
    CUTE_INVALID_CONTROL_PATH("Cannot use Xe DPAS MMA atom on non-Xe hardware"); \
  } \
};
#endif


#define CUTE_DECLARE_XE_DPAS_TT_ALLM(TD, TA, TB, TC)  \
  CUTE_DECLARE_XE_DPAS_TT(1, 8,  16,  TD, TA, TB, TC) \
  CUTE_DECLARE_XE_DPAS_TT(2, 16, 32,  TD, TA, TB, TC) \
  CUTE_DECLARE_XE_DPAS_TT(4, 32, 64,  TD, TA, TB, TC) \
  CUTE_DECLARE_XE_DPAS_TT(8, 64, 128, TD, TA, TB, TC)

CUTE_DECLARE_XE_DPAS_TT_ALLM(f,   tf32, tf32, f)

CUTE_DECLARE_XE_DPAS_TT_ALLM(f,   bf,   bf,   f)
CUTE_DECLARE_XE_DPAS_TT_ALLM(bf,  bf,   bf,   f)
CUTE_DECLARE_XE_DPAS_TT_ALLM(f,   bf,   bf,   bf)
CUTE_DECLARE_XE_DPAS_TT_ALLM(bf,  bf,   bf,   bf)

CUTE_DECLARE_XE_DPAS_TT_ALLM(f,   hf,   hf,   f)
CUTE_DECLARE_XE_DPAS_TT_ALLM(f,   hf,   hf,   hf)
CUTE_DECLARE_XE_DPAS_TT_ALLM(hf,  hf,   hf,   f)
CUTE_DECLARE_XE_DPAS_TT_ALLM(hf,  hf,   hf,   hf)

CUTE_DECLARE_XE_DPAS_TT_ALLM(ud,  u8,   u8,   ud)
CUTE_DECLARE_XE_DPAS_TT_ALLM(d,   u8,   u8,   d)
CUTE_DECLARE_XE_DPAS_TT_ALLM(d,   u8,   s8,   d)
CUTE_DECLARE_XE_DPAS_TT_ALLM(d,   s8,   u8,   d)
CUTE_DECLARE_XE_DPAS_TT_ALLM(d,   s8,   s8,   d)

} //namespace cute
