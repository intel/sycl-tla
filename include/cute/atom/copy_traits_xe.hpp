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
#pragma once

#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits.hpp>

#include <cute/arch/copy_xe.hpp>

namespace cute
{
template <class CopyOp, class... ArgTs> struct XE_2D_LD_Unpack {
  const void *base_ptr;
  uint32_t width;
  uint32_t height;
  uint32_t pitch;

  XE_2D_LD_Unpack(const void *ptr, uint32_t const &w,
                  uint32_t const &h, uint32_t const &p)
      : base_ptr(ptr), width(w), height(h), pitch(p) {}

  template <class TraitsArgs>
  XE_2D_LD_Unpack(TraitsArgs const &traits) : base_ptr(traits.base_ptr),
              width(traits.width), height(traits.height), pitch(traits.pitch) {}

  XE_2D_LD_Unpack() {}

  using Traits_LD_t = Copy_Traits<CopyOp, ArgTs...>;

  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Traits_LD_t const &traits, Tensor<TS, SLayout> const &src,
              Tensor<TD, DLayout> &dst) {
    static_assert(is_rmem<TD>::value);

    using dtype = typename Tensor<TD, DLayout>::value_type;

    dtype *base_addr = (dtype *)traits.base_ptr;

    auto [m, n, l] = src.data().coord_;

    CopyOp::copy(base_addr + l * traits.width * traits.height,
                 traits.width * sizeof(dtype), traits.height,
                 traits.pitch * sizeof(dtype), intel::coord_t{(int)n, (int)m},
                 &*dst.data());
  }

  template <class GCoord, class GShape, class GStride>
  CUTE_HOST_DEVICE constexpr auto get_pvc_tensor(GCoord const &coord,
                                                 GShape const &shape,
                                                 GStride const &stride) const {
    auto t_shape = make_shape(_1{}, get<1>(shape), get<2>(shape));
    auto t_stride = make_stride(_1{}, get<0>(stride) * CopyOp::blk_height,
                                get<1>(stride) * CopyOp::blk_width);
    return make_tensor(make_inttuple_iter(coord),
                       make_layout(t_shape, t_stride));
  }

  template <class T1, class T2, class... TraitsArgs>
  static constexpr auto with(T1 && arg1, T2 && arg2, TraitsArgs&&... args) {
      return Traits_LD_t{arg1, arg2, args...};
  }
};

template <class CopyOp, class... ArgTs> struct XE_2D_ST_Unpack {
  const void *base_ptr;
  uint32_t width;
  uint32_t height;
  uint32_t pitch;

  XE_2D_ST_Unpack(const void *ptr, uint32_t const &w,
                  uint32_t const &h, uint32_t const &p)
      : base_ptr(ptr), width(w), height(h), pitch(p) {}

  template <class TraitsArgs>
  XE_2D_ST_Unpack(TraitsArgs const &traits)  : base_ptr(traits.base_ptr),
                  width(traits.width), height(traits.height), pitch(traits.pitch) {}

  XE_2D_ST_Unpack() {}

  using Traits_ST_t = Copy_Traits<CopyOp, ArgTs...>;

  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Traits_ST_t const &traits, Tensor<TS, SLayout> const &src,
              Tensor<TD, DLayout> &dst) {
    static_assert(is_rmem<TS>::value);

    using dtype = typename Tensor<TS, SLayout>::value_type;

    dtype *base_addr = (dtype *)traits.base_ptr;

    auto [m, n, l] = dst.data().coord_;

    CopyOp::copy(base_addr + l * traits.width * traits.height,
                 (int)(traits.width * sizeof(dtype)), (int)(traits.height),
                 (int)(traits.pitch * sizeof(dtype)),
                 intel::coord_t{(int)n, (int)m}, &*src.data());
  }

  template <class GCoord, class GShape>
  CUTE_HOST_DEVICE constexpr auto get_pvc_tensor(GCoord const &coord,
                                                 GShape const &shape) const {
    auto t_shape = make_shape(_1{}, get<1>(shape), get<2>(shape));
    auto t_stride = make_stride(_1{}, E<0>{} * CopyOp::blk_height,
                                E<1>{} * CopyOp::blk_width);
    return make_tensor(make_inttuple_iter(coord),
                       make_layout(t_shape, t_stride));
  }

  template <class T1, class T2, class... TraitsArgs>
  static constexpr auto with(T1 && arg1, T2 && arg2, TraitsArgs&&... args) {
      return Traits_ST_t{arg1, arg2, args...};
  }
};

template <class CopyOp, class GTensor> struct XE_2D_PF_Unpack {
  GTensor tensor;

  template <class G_Tensor>
  CUTE_HOST_DEVICE XE_2D_PF_Unpack(G_Tensor const &t) : tensor(t) {}

  using Copy_Traits = Copy_Traits<CopyOp, GTensor>;
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void copy_unpack(
      Copy_Traits const &traits,
      Tensor<ViewEngine<ArithmeticTupleIterator<TS>>, SLayout> const &src,
      Tensor<TD, DLayout> &dst) {
    using T = typename Copy_Traits::CopyInternalType;
    int H = size<0>(traits.tensor);
    int W = size<1>(traits.tensor) * sizeof(T);
    auto [y, x, z] = src.data().coord_;
    CopyOp::template copy<T>(
        traits.tensor.data() + z * W * H / sizeof(T), W, H, W,
        intel::coord_t{static_cast<int>(x), static_cast<int>(y)});
  }
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x1x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x1x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_16>,
                           Stride<_16, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x1x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x2x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x2x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x2x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x2x32_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U8x2x32_ST_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout =
      Layout<Shape<_16, Shape<_2, _16>>, Stride<_0, Stride<_0, _1>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = SrcLayout;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  using CopyInternalType = uint8_t;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U8x2x32_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x4x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x4x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _4>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x4x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x16x8_LD_T, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x16x8_LD_T, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_8, _16>>,
                           Stride<_16,Stride< _0,  _1>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = SrcLayout;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U16x16x8_LD_T, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x16x16_LD_T, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x16x16_LD_T, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16, _16>>,
                           Stride<_16,Stride< _0,  _1>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = SrcLayout;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U16x16x16_LD_T, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x8x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x8x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _8>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x8x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x16x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x16x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16, _16>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x16x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x32x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x32x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16, _32>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x32x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x1x64_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x1x64_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x1x64_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x2x64_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x2x64_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _2>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x2x64_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x4x64_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x4x64_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _4>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x4x64_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x8x64_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x8x64_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _8>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U8x8x64_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x16x64_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x16x64_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2, _16>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x32x64_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x32x64_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2, _32>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x1x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x1x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_16>,
                           Stride<_16, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U16x1x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x2x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x2x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U16x2x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x4x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x4x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _4>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_LD_Unpack<XE_2D_U16x4x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x8x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x8x16_LD_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_256, Shape <_8, _16>>,
                           Stride< _1, Stride<_256,  _1>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = SrcLayout;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x8x16_LD_N, args_t...>(args...) {}
};

template <class GTensor>
struct Copy_Traits<XE_2D_U16x8x16_LD_N::PREFETCH, GTensor>
    : XE_2D_PF_Unpack<XE_2D_U16x8x16_LD_N::PREFETCH, GTensor> {
  //   template <class... CopyArgs>
  //   CUTE_HOST_DEVICE Copy_Traits(Copy_Traits<CopyArgs...> const &traits)
  //       : XE_2D_PF_Unpack<XE_2D_U16x8x16_LD_N::PREFETCH,
  //       GTensor>(traits.tensor) {
  //   }

  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16,  _8>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _8>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  using CopyInternalType = ushort;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x16x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x16x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16, _16>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x16x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x32x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x32x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16, _32>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x32x16_LD_N, args_t...>(args...) {}
};

template <class GTensor>
struct Copy_Traits<XE_2D_U16x32x16_LD_N::PREFETCH, GTensor>
    : XE_2D_PF_Unpack<XE_2D_U16x32x16_LD_N::PREFETCH, GTensor> {

  //   template <class... CopyArgs>
  //   CUTE_HOST_DEVICE Copy_Traits(Copy_Traits<CopyArgs...> const &traits)
  //       : XE_2D_PF_Unpack<XE_2D_U16x32x16_LD_N::PREFETCH, GTensor>(
  //             traits.tensor) {}

  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16, _32>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16, _32>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  using CopyInternalType = ushort;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x1x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x1x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x1x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x2x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x2x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _2>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x2x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x4x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x4x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _4>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x4x32_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x8x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x8x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _8>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x8x32_LD_N, args_t...>(args...) {}
};

template <class GTensor>
struct Copy_Traits<XE_2D_U16x8x32_LD_N::PREFETCH, GTensor>
    : XE_2D_PF_Unpack<XE_2D_U16x8x32_LD_N::PREFETCH, GTensor> {

  //   template <class... CopyArgs>
  //   CUTE_HOST_DEVICE Copy_Traits(Copy_Traits<CopyArgs...> const &traits)
  //       : XE_2D_PF_Unpack<XE_2D_U16x8x32_LD_N::PREFETCH,
  //       GTensor>(traits.tensor) {
  //   }

  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_32,  _8>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32,  _8>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  using CopyInternalType = ushort;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x16x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x16x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2, _16>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x16x32_LD_N, args_t...>(args...) {}
};

template <class GTensor>
struct Copy_Traits<XE_2D_U16x16x32_LD_N::PREFETCH, GTensor>
    : XE_2D_PF_Unpack<XE_2D_U16x16x32_LD_N::PREFETCH, GTensor> {
  //   template <class... CopyArgs>
  //   CUTE_HOST_DEVICE Copy_Traits(Copy_Traits<CopyArgs...> const &traits)
  //       : XE_2D_PF_Unpack<XE_2D_U16x16x32_LD_N::PREFETCH, GTensor>(
  //             traits.tensor) {}

  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_32, _16>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32, _16>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  using CopyInternalType = ushort;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x32x32_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x32x32_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2, _32>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x32x32_LD_N, args_t...>(args...) {}
};

template <class GTensor>
struct Copy_Traits<XE_2D_U16x32x32_LD_N::PREFETCH, GTensor>
    : XE_2D_PF_Unpack<XE_2D_U16x32x32_LD_N::PREFETCH, GTensor> {

  //   template <class... CopyArgs>
  //   CUTE_HOST_DEVICE Copy_Traits(Copy_Traits<CopyArgs...> const &traits)
  //       : XE_2D_PF_Unpack<XE_2D_U16x32x32_LD_N::PREFETCH, GTensor>(
  //             traits.tensor) {}

  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_32, _32>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32, _32>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  using CopyInternalType = ushort;
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x1x8_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x1x8_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,_2>,_32>,
                           Stride<Stride<_32,_0>, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x1x8_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x2x8_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x2x8_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,_32>,
                           Stride<Stride<_32,_256>, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x2x8_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x4x8_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x4x8_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32,  _2>>,
                           Stride<Stride<_32,_256>,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x4x8_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x8x8_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x8x8_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32,  _4>>,
                           Stride<Stride<_32,_256>,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x8x8_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x16x8_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x16x8_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32,   _8>>,
                           Stride<Stride<_32,_256>,Stride< _1, _512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x16x8_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x32x8_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x32x8_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32, _16>>,
                           Stride<Stride<_32,_256>,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x32x8_LD_N, args_t...>(args...) {}
};
template <class... args_t>
struct Copy_Traits<XE_2D_TF32x1x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x1x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,_2>,Shape <_32,  _2>>,
                           Stride<Stride<_32,_0>,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x1x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x2x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x2x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32,  _2>>,
                           Stride<Stride<_32,_512>,Stride< _1,_256>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x2x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x4x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x4x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32,  _2,   _2>>,
                           Stride<Stride<_32,_512>,Stride< _1,_256,_1024>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x4x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x8x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x8x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32,  _2,   _4>>,
                           Stride<Stride<_32,_512>,Stride< _1,_256,_1024>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x8x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x16x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x16x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32,  _2,  _16>>,
                           Stride<Stride<_32,_512>,Stride< _1,_256,_1024>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x16x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_TF32x32x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_TF32x32x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,  _2>,Shape <_32,  _2,  _32>>,
                           Stride<Stride<_32,_512>,Stride< _1,_256,_1024>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_TF32x32x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x1x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x1x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_32>,
                           Stride<_32, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x1x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x2x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x2x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32,  _2>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x2x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x4x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x4x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32,  _4>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x4x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x8x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x8x16_LD_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32,  _8>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  static constexpr auto blk_width = 1;
  static constexpr auto blk_height = 8;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x8x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x16x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x16x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32, _16>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x16x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x32x16_LD_N, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x32x16_LD_N, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32, _32>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x32x16_LD_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x32x16_LD_V, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x32x16_LD_V, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_8,  _4,  _8>>,
                           Stride< _8,Stride<_1,_128,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U8x32x16_LD_V, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x32x32_LD_V, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x32x32_LD_V, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_8,  _4,  _2,  _8>>,
                           Stride< _8,Stride<_1,_256,_128,_1024>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U8x32x32_LD_V, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x32x64_LD_V, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U8x32x64_LD_V, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_8,  _4,  _4,   _8>>,
                           Stride< _8,Stride<_1,_512,_128,_2048>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U8x32x64_LD_V, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x16x16_LD_V, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x16x16_LD_V, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_256, Shape < _16,_16>>,
                           Stride<  _1, Stride<_256, _1>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = SrcLayout;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x16x16_LD_V, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x32x16_LD_V, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x32x16_LD_V, args_t...> {
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2, _16>>,
                           Stride<_16,Stride< _1,_256,_512>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x32x16_LD_V, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x32x32_LD_V, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x32x32_LD_V, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_64>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _2,  _16>>,
                           Stride<_16,Stride< _1,_512,_256,_1024>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x32x32_LD_V, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x16x32_LD_V, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U16x16x32_LD_V, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_64>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_16,  _2,  _2,   _8>>,
                           Stride<_16,Stride< _1,_512,_256,_1024>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U16x16x32_LD_V, args_t...>(args...) {}
};

// template<class... args_t>
// struct Copy_Traits<XE_2D_U32x16x1_LD_T, args_t...>
//     : XE_2D_LD_Unpack<XE_2D_U32x16x1_LD_T, args_t...> {
//   // Logical thread id to thread idx
//   using ThrID = Layout<_16>;
//   // Map from (src-thr,src-val) to bit
//   using SrcLayout = Layout<Shape <_16,_32>,
//                            Stride< _0, _1>>;
//   // Map from (dst-thr,dst-val) to bit
//   using DstLayout = Layout<Shape <_16,_32>,
//                            Stride<_32, _1>>;
//   // Reference map from (thr,val) to bit
//   using RefLayout = DstLayout;
// };

template <class... args_t>
struct Copy_Traits<XE_2D_U32x16x2_LD_T, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x16x2_LD_T, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32, _2>>,
                           Stride<_64,Stride< _1,_32>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x16x2_LD_T, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x16x4_LD_T, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x16x4_LD_T, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,Shape <_32, _4>>,
                           Stride<_64,Stride< _1,_32>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x16x4_LD_T, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x16x8_LD_T, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U32x16x8_LD_T, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape < _16,Shape <_32, _8>>,
                           Stride<_128,Stride< _1,_32>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U32x16x8_LD_T, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U64x8x1_LD_T, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U64x8x1_LD_T, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_64>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape < _8,_2>,_64>,
                           Stride<Stride<_64,_0>, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U64x8x1_LD_T, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U64x8x2_LD_T, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U64x8x2_LD_T, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_64>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape <  _8,_2>,Shape <_64, _2>>,
                           Stride<Stride<_128,_0>,Stride< _1,_64>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U64x8x2_LD_T, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U64x8x4_LD_T, args_t...>
    : XE_2D_LD_Unpack<XE_2D_U64x8x4_LD_T, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_64>,
                           Stride< _0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <Shape <  _8,_2>,Shape <_64, _4>>,
                           Stride<Stride<_256,_0>,Stride< _1,_64>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_LD_Unpack<XE_2D_U64x8x4_LD_T, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x1x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U8x1x16_ST_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_8>,
                           Stride< _8,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_ST_Unpack<XE_2D_U8x1x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x2x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U8x2x16_ST_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_8,  _2>>,
                           Stride< _8,Stride<_1,_128>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_ST_Unpack<XE_2D_U8x2x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x4x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U8x4x16_ST_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_8,  _4>>,
                           Stride< _8,Stride<_1,_128>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x8x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U8x8x16_ST_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_8,  _8>>,
                           Stride< _8,Stride<_1,_128>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
  using CopyInternalType = uint8_t;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U8x8x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U8x8x32_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U8x8x32_ST_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout =
      Layout<Shape<_16, Shape<_8, _16>>, Stride<_0, Stride<_0, _1>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_8>,
                           Stride< _0,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
  using CopyInternalType = uint8_t;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U8x8x32_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x1x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U16x1x16_ST_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_16>,
                           Stride<_16, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U16x1x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x2x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U16x2x16_ST_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16,  _2>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U16x2x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x4x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U16x4x16_ST_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16,  _4>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U16x4x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U16x8x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U16x8x16_ST_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_16,  _8>>,
                           Stride<_16,Stride< _1,_256>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_16>,
                           Stride< _0, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U16x8x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x1x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U32x1x16_ST_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,_32>,
                           Stride<_32, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U32x1x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x2x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U32x2x16_ST_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_32,  _2>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U32x2x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x4x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U32x4x16_ST_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_16,Shape <_32,  _4>>,
                           Stride<_32,Stride< _1,_512>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_16,_32>,
                           Stride< _0, _1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  template <class... ArgT>
  Copy_Traits(ArgT... args)
      : XE_2D_ST_Unpack<XE_2D_U32x4x16_ST_N, args_t...>(args...) {}
};

template <class... args_t>
struct Copy_Traits<XE_2D_U32x8x16_ST_N, args_t...>
    : XE_2D_ST_Unpack<XE_2D_U32x8x16_ST_N, args_t...> {
  // Logical thread id to thread idx
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <_512, Shape <  _8,_32>>,
                           Stride<  _1, Stride<_512, _1>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = SrcLayout;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  template <class... ArgTs>
  Copy_Traits(ArgTs... args)
      : XE_2D_ST_Unpack<XE_2D_U32x8x16_ST_N, args_t...>(args...) {}
};

template <class Copy, class GEngine, class GLayout>
auto make_xe_2d_copy(Tensor<GEngine, GLayout> gtensor) {
  using GTensor = Tensor<GEngine, GLayout>;
  using Traits = Copy_Traits<Copy, GTensor>;
  // Traits traits {gtensor};
  return Copy_Atom<Traits, typename GEngine::value_type>{gtensor};
}

template<class S, class D>
struct Copy_Traits<XE_ATOMIC<S, D>> {
  // Logical thread id to thread idx (one-thread)
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,Int<sizeof_bits<S>::value>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,Int<sizeof_bits<D>::value>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

template<class S, class D>
struct Copy_Traits<XE_1D_LDSM<S, D>> {
    // Logical thread id to thread idx
    using ThrID = Layout<_16>;
    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_16, Int<sizeof_bits<S>::value>>, Stride<_0, _1>>;
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_16, Shape<Int<sizeof_bits<D>::value / sizeof_bits<S>::value>, Int<sizeof_bits<S>::value>>>,
                             Stride<Int<sizeof_bits<S>::value>, Stride<Int<sizeof_bits<S>::value * 16>, _1>>>;
    // Reference map from (thr,val) to bit
    using RefLayout = DstLayout;
};

template<class S, class D>
struct Copy_Traits<XE_1D_LOAD_GLOBAL<S, D>> {
    // Logical thread id to thread idx
    using ThrID = Layout<_16>;
    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_16, Int<sizeof_bits<S>::value>>, Stride<_0, _1>>;
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape <_16, Shape<Int<sizeof_bits<D>::value / sizeof_bits<S>::value>, Int<sizeof_bits<S>::value>>>,
                             Stride<Int<sizeof_bits<S>::value>, Stride<Int<sizeof_bits<S>::value * 16>, _1>>>;
    // Reference map from (thr,val) to bit
    using RefLayout = DstLayout;
};

template<class S, class D>
struct Copy_Traits<PREFETCH<S, D>> {
    // Logical thread id to thread idx
    using ThrID = Layout<_16>;
    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_16, Int<sizeof_bits<S>::value>>, Stride<_0, _1>>;
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape <                       _16,Int<sizeof_bits<D>::value>>,
                             Stride<Int<sizeof_bits<D>::value>,                        _1>>;
    // Reference map from (thr,val) to bit
    using RefLayout = DstLayout;

    template <class... CopyArgs>
    CUTE_HOST_DEVICE
    Copy_Traits(Copy_Traits<CopyArgs...> const& traits) {}
};

template<class S, class D>
struct Copy_Traits<XE_1D_STSM<S, D>> {
    // Logical thread id to thread idx
    using ThrID = Layout<_16>;
    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_16, Shape<Int<sizeof_bits<S>::value / sizeof_bits<D>::value>, Int<sizeof_bits<D>::value>>>,
                             Stride<Int<sizeof_bits<D>::value>, Stride<Int<sizeof_bits<D>::value * 16>, _1>>>;
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_16, Int<sizeof_bits<D>::value>>, Stride<_0, _1>>;
    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
};

template<class S, class D>
struct Copy_Traits<XE_1D_STORE_GLOBAL<S, D>> {
    // Logical thread id to thread idx
    using ThrID = Layout<_16>;
    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_16, Shape<Int<sizeof_bits<S>::value / sizeof_bits<D>::value>, Int<sizeof_bits<D>::value>>>,
                             Stride<Int<sizeof_bits<D>::value>, Stride<Int<sizeof_bits<D>::value * 16>, _1>>>;
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_16, Int<sizeof_bits<D>::value>>, Stride<_0, _1>>;
    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
};

} // end namespace cute
