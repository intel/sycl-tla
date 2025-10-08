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
  Shape rshape = replace<Mode>(shape, _1{});
  Tensor out = make_subgroup_tensor(make_tensor<T>(ceil_div(size(rshape), intel::_SGSize{})),
                                    make_identity_layout(rshape));

  /* Check for reduction type */
  constexpr bool horizontal = (size<0>(rcoord_to_tv) == intel::_SGSize{} * size<0>(rcoord_to_v));
  constexpr bool vertical   = (size<1>(rcoord_to_tv) == intel::_SGSize{} * size<1>(rcoord_to_v));

  CUTE_UNROLL
  for (int j = 0; j < size<1>(rcoord_to_v); j++) {
    T acc = src_r(0, j);
    CUTE_UNROLL
    for (int i = 1; i < size<0>(rcoord_to_v); i++) {
      acc = op(acc, src_r(i, j));
    }

    if constexpr (horizontal)
      set_single_value(out, j, reduce_over_group(sg, acc, op));   // TODO: optimize vector usage
    else if constexpr (vertical)
      out(j) = acc;
    else
      static_assert("Unimplemented reduction type");
  }

  return out;
}

} // namespace cute
