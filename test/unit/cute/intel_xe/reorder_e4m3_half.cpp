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

// Xe2 (BMG) e4m3 -> half reorder atoms, default path: every e4m3 code, exact (NaN -> NaN).
// The CUTE_XE_E4M3_HALF_UNSCALED path is tested in reorder_e4m3_half_unscaled.cpp (separate
// executable: the define changes the same inline atoms, so it cannot share a binary with this file).

#include "reorder_e4m3_half.hpp"

#if !(defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))

template <class...> class E4m3HalfDefaultKernel;

TEST(BMG_CuTe_Xe_Reorder_E4m3_Half, uu_default_all_codes) {
  e4m3_half_test::run_all_codes<cute::ReorderKind::UU, false, E4m3HalfDefaultKernel<cute::_0>>();
}

TEST(BMG_CuTe_Xe_Reorder_E4m3_Half, vv_default_all_codes) {
  e4m3_half_test::run_all_codes<cute::ReorderKind::VV, false, E4m3HalfDefaultKernel<cute::_1>>();
}

#endif
