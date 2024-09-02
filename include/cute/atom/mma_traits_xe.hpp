#pragma once

#include <cute/arch/mma_xe.hpp>
#include <cute/atom/mma_traits.hpp>

#include <cute/layout.hpp>

namespace cute
{
template <>
struct MMA_Traits<XE_8x16x16_BF16BF16F32F32_NN>
{
  using ValTypeD = float;
  using ValTypeA = sycl::ext::oneapi::bfloat16;
  using ValTypeB = sycl::ext::oneapi::bfloat16;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_16,_16>;
  using ThrID   = Layout<_16>;

  using ALayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
  using BLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
};

template <>
struct MMA_Traits<XE_4x16x16_F32BF16BF16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_4,_16,_16>;
  using ThrID   = Layout<_16>;

  using ALayout = Layout<Shape<_16, _4>, Stride<_4, _1>>;
  using BLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _4>, Stride<_4, _1>>;
};

template <>
struct MMA_Traits<XE_2x16x16_F32BF16BF16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_2,_16,_16>;
  using ThrID   = Layout<_16>;

  using ALayout = Layout<Shape<_16, _2>, Stride<_2, _1>>;
  using BLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _2>, Stride<_2, _1>>;
};

template <>
struct MMA_Traits<XE_1x16x16_F32BF16BF16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_1,_16,_16>;
  using ThrID   = Layout<_16>;

  using ALayout = Layout<Shape<_16, _1>, Stride<_1, _1>>;
  using BLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _1>, Stride<_1, _1>>;
};


template <>
struct MMA_Traits<XE_8x16x16_F32F16F16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_16,_16>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
  using BLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
};

template <>
struct MMA_Traits<XE_4x16x16_F32F16F16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_4,_16,_16>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, _4>, Stride<_4, _1>>;
  using BLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _4>, Stride<_4, _1>>;
};

template <>
struct MMA_Traits<XE_2x16x16_F32F16F16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_2,_16,_16>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, _2>, Stride<_2, _1>>;
  using BLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _2>, Stride<_2, _1>>;
};

template <>
struct MMA_Traits<XE_1x16x16_F32F16F16F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_1,_16,_16>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, _1>, Stride<_1, _1>>;
  using BLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _1>, Stride<_1, _1>>;
};

template <>
struct MMA_Traits<XE_8x16x32_S32S8S8S32_TT>
{
  using ValTypeD = int;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int;

  using Shape_MNK = Shape<_8,_16,_32>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, Shape<_2, _8>>, Stride<_16, Stride<_8, _1>>>;
  using BLayout = Layout<Shape<_16, _32>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
};

template <>
struct MMA_Traits<XE_4x16x32_S32S8S8S32_TT>
{
  using ValTypeD = int;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int;

  using Shape_MNK = Shape<_4,_16,_32>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, Shape<_2, _4>>, Stride<_8, Stride<_4, _1>>>;
  using BLayout = Layout<Shape<_16, _32>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _4>, Stride<_4, _1>>;
};

template <>
struct MMA_Traits<XE_2x16x32_S32S8S8S32_TT>
{
  using ValTypeD = int;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int;

  using Shape_MNK = Shape<_2,_16,_32>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, Shape<_2, _2>>, Stride<_4, Stride<_2, _1>>>;
  using BLayout = Layout<Shape<_16, _32>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _2>, Stride<_2, _1>>;
};

template <>
struct MMA_Traits<XE_1x16x32_S32S8S8S32_TT>
{
  using ValTypeD = int;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int;

  using Shape_MNK = Shape<_1,_16,_32>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, Shape<_2, _1>>, Stride<_2, Stride<_1, _1>>>;
  using BLayout = Layout<Shape<_16, _32>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _1>, Stride<_1, _1>>;
};

template <>
struct MMA_Traits<XE_8x16x32_S32U8U8S32_TT>
{
  using ValTypeD = int;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int;

  using Shape_MNK = Shape<_8,_16,_32>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, Shape<_2, _8>>, Stride<_16, Stride<_8, _1>>>;
  using BLayout = Layout<Shape<_16, _32>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
};

template <>
struct MMA_Traits<XE_4x16x32_S32U8U8S32_TT>
{
  using ValTypeD = int;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int;

  using Shape_MNK = Shape<_4,_16,_32>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, Shape<_2, _4>>, Stride<_8, Stride<_4, _1>>>;
  using BLayout = Layout<Shape<_16, _32>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _4>, Stride<_4, _1>>;
};

template <>
struct MMA_Traits<XE_2x16x32_S32U8U8S32_TT>
{
  using ValTypeD = int;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int;

  using Shape_MNK = Shape<_2,_16,_32>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, Shape<_2, _2>>, Stride<_4, Stride<_2, _1>>>;
  using BLayout = Layout<Shape<_16, _32>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _2>, Stride<_2, _1>>;
};

template <>
struct MMA_Traits<XE_1x16x32_S32U8U8S32_TT>
{
  using ValTypeD = int;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int;

  using Shape_MNK = Shape<_1,_16,_32>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<_16, Shape<_2, _1>>, Stride<_2, Stride<_1, _1>>>;
  using BLayout = Layout<Shape<_16, _32>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _1>, Stride<_1, _1>>;
};

template <>
struct MMA_Traits<XE_8x16x8_F32TF32TF32F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_8,_16,_8>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<Shape<_8, _2>, _4>, Stride<Stride<_8, _1>, _2>>;
  using BLayout = Layout<Shape<_16, _8>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
};

template <>
struct MMA_Traits<XE_4x16x8_F32TF32TF32F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_4,_16,_8>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<Shape<_8, _2>, _2>, Stride<Stride<_4, _1>, _2>>;
  using BLayout = Layout<Shape<_16, _8>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _4>, Stride<_4, _1>>;
};

template <>
struct MMA_Traits<XE_2x16x8_F32TF32TF32F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_2,_16,_8>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<Shape<_8, _2>, _1>, Stride<Stride<_2, _1>, _0>>;
  using BLayout = Layout<Shape<_16, _8>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _2>, Stride<_2, _1>>;
};

template <>
struct MMA_Traits<XE_1x16x8_F32TF32TF32F32_TT>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_1,_16,_8>;
  using ThrID   = Layout<_16>;
  using ALayout = Layout<Shape<Shape<_8, _2>, _1>, Stride<Stride<_1, _0>, _0>>;
  using BLayout = Layout<Shape<_16, _8>, Stride<_1, _16>>;
  using CLayout = Layout<Shape<_16, _1>, Stride<_1, _1>>;
};

}
